import functools
import os
import struct
from collections import defaultdict

import math

from pacman.exceptions import PacmanAlgorithmFailedToCompleteException
from pacman.model.routing_tables import MulticastRoutingTables, \
    UnCompressedMulticastRoutingTable
from pacman.operations.algorithm_reports.reports import format_route
from pacman.operations.router_compressors.mundys_router_compressor.\
    routing_table_condenser import MundyRouterCompressor
from spinn_machine import MulticastRoutingEntry
from spinn_utilities.default_ordered_dict import DefaultOrderedDict
from spinn_utilities.find_max_success import find_max_success
from spinn_utilities.progress_bar import ProgressBar
from spynnaker.pyNN.models.abstract_models.\
    abstract_uses_bit_field_filterer import AbstractUsesBitFieldFilter
from rig.routing_table import ordered_covering as rigs_compressor, \
    MinimisationFailedError

from spynnaker.pyNN.utilities import constants


class HostBasedBitFieldRouterCompressor(object):
    """ host based fancy router compressor using the bitfield filters of the \
    cores.
    """

    __slots__ = [
        "_cached_entries",
        "_last_successful",
        "_last_successful_bit_fields_merged"
    ]

    # max entries that can be used by the application code
    MAX_SUPPORTED_LENGTH = 1023

    # report name
    REPORT_NAME = "router_compressor_with_bitfield.rpt"

    # bytes per word
    _BYTES_PER_WORD = 4

    # key id for the initial entry
    ORIGINAL_ENTRY = 0

    # key id for the bitfield entries
    ENTRIES = 1

    # bits in a word
    _BITS_IN_A_WORD = 32

    # bit to mask a bit
    _BIT_MASK = 1

    # mask for neuron level
    NEURON_LEVEL_MASK = 0xFFFFFFFF

    # structs for performance requirements.
    _ONE_WORDS = struct.Struct("<I")
    _TWO_WORDS = struct.Struct("<II")
    _FOUR_WORDS = struct.Struct("<IIII")

    # for router report
    _LOWER_16_BITS = 0xFFFF

    def __init__(self):
        self._cached_entries = dict()
        self._last_successful = None
        self._last_successful_bit_fields_merged = None

    def __call__(
            self, router_tables, machine, placements, transceiver,
            graph_mapper, default_report_folder, target_length=None):
        """ compresses bitfields and router table entries together as /
        feasible as possible
        
        :param router_tables: routing tables (uncompressed)
        :param machine: SpiNNMachine instance
        :param placements: placements
        :param transceiver: SpiNNMan instance
        :param graph_mapper: mapping between graphs
        :param default_report_folder: report folder
        :param target_length: length of table entries to get to.
        :return: compressed routing table entries
        """

        # create progress bar
        progress = ProgressBar(
            len(router_tables.routing_tables) * 2,
            "Compressing routing Tables with bitfields in host")

        # create report
        report_file_path = os.path.join(default_report_folder, self.REPORT_NAME)
        report_out = open(report_file_path, "w")

        # holder for the bitfields in
        bit_field_sdram_base_addresses = defaultdict(dict)

        # compressed router table
        compressed_pacman_router_tables = MulticastRoutingTables()

        # locate the bitfields in a chip level scope
        for router_table in progress.over(router_tables.routing_tables, False):
            n_processors_on_chip = machine.get_chip_at(
                router_table.x, router_table.y).n_processors
            for processor_id in range(0, n_processors_on_chip):
                if placements.is_processor_occupied(
                        router_table.x, router_table.y, processor_id):
                    machine_vertex = placements.get_vertex_on_processor(
                        router_table.x, router_table.y, processor_id)
                    app_vertex = graph_mapper.get_application_vertex(
                        machine_vertex)
                    if isinstance(app_vertex, AbstractUsesBitFieldFilter):
                        bit_field_sdram_base_addresses[
                            (router_table.x, router_table.y)][processor_id] = \
                            app_vertex.bit_field_base_address(
                                transceiver,
                                placements.get_placement_of_vertex(
                                    machine_vertex))

        # start the routing table choice conversion
        for router_table in router_tables.routing_tables:

            # iterate through bitfields on this chip and convert to router table
            bit_field_chip_base_addresses = bit_field_sdram_base_addresses[
                (router_table.x, router_table.y)]

            # read in bitfields.
            bit_fields_by_processor, bit_field_by_key = \
                self._read_in_bit_fields(
                    transceiver, router_table.x, router_table.y,
                    bit_field_chip_base_addresses)

            # execute binary search
            self._start_binary_search(
                router_table, bit_field_by_key, target_length)

            # add final to compressed tables
            compressed_pacman_router_tables.add_routing_table(
                self._last_successful)

            # remove bitfields from cores that have been merged into the router\
            # table
            self._remove_merged_bitfields_from_cores(
                self._last_successful_bit_fields_merged, router_table.x,
                router_table.y, transceiver,
                bit_field_chip_base_addresses, bit_fields_by_processor)

            # report
            self._record_changes(
                router_table, self._last_successful,
                self._last_successful_bit_fields_merged, report_out)

        # clean report
        report_out.flush()
        report_out.close()

        # return compressed tables
        return compressed_pacman_router_tables

    def _convert_bitfields_into_router_tables(
            self, router_table, bitfields_by_key):
        """ converts the bitfield into router table entries for compression. \
        based off the entry located in the original router table
        
        :param router_table: the original routing table
        :param bitfields_by_key: the bitfields of the chip.
        :return: routing tables . 
        """
        bit_field_router_tables = list()

        # clone the original entries
        original_route_entries = list()
        original_route_entries.extend(router_table.multicast_routing_entries)

        # go through the bitfields and get the routing table for it
        for master_pop_key in bitfields_by_key.keys():

            # if not cached, generate
            if master_pop_key not in self._cached_entries:
                self._cached_entries[master_pop_key] = dict()

                # store the original entry that's going to be deleted from this
                self._cached_entries[master_pop_key][self.ORIGINAL_ENTRY] = \
                    router_table.get_entry_by_routing_entry_key(master_pop_key)

                # store the bitfield routing table
                self._cached_entries[master_pop_key][self.ENTRIES] = \
                    UnCompressedMulticastRoutingTable(
                        router_table.x, router_table.y,
                        multicast_routing_entries=(
                            self._generate_entries_from_bitfield(
                                bitfields_by_key[master_pop_key],
                                self._cached_entries[master_pop_key][
                                    self.ORIGINAL_ENTRY])))

            # add to the list
            bit_field_router_tables.append(
                self._cached_entries[master_pop_key][self.ENTRIES])

            # remove entry
            original_route_entries.remove(
                self._cached_entries[master_pop_key][self.ORIGINAL_ENTRY])

        # create reduced
        reduced_original_table = UnCompressedMulticastRoutingTable(
            router_table.x, router_table.y, original_route_entries)

        # add reduced to front of the tables
        bit_field_router_tables.insert(0, reduced_original_table)

        # return the bitfield tables and the reduced original table
        return bit_field_router_tables

    def _generate_entries_from_bitfield(self, bit_fields, routing_table_entry):
        """ generate neuron level entries 
        
        :param bit_fields: the bitfields for a given key
        :param routing_table_entry: the original entry from it
        :return: the set of bitfield entries
        """
        entries = list()

        # get some basic values
        entry_links = routing_table_entry.link_ids
        base_key = routing_table_entry.routing_entry_key
        (bit_field, processor_id) = bit_fields[0]
        n_neurons = len(bit_field) * self._BITS_IN_A_WORD

        # check each neuron to see if any bitfields care, and if so,
        # add processor
        for neuron in range(0, n_neurons):
            processors = list()
            for (bit_field, processor_id) in bit_fields:
                if self._bit_for_neuron_id(bit_field, neuron):
                    processors.append(processor_id)
            entries.append(MulticastRoutingEntry(
                routing_entry_key=base_key + neuron,
                mask=self.NEURON_LEVEL_MASK,
                link_ids=entry_links,
                defaultable=False,
                processor_ids=processors))

        # return the entries
        return entries

    def _bit_for_neuron_id(self, bit_field, neuron_id):
        """ locate the bit for the neuron in the bitfield

        :param bit_field: the block of words which represent the bitfield
        :param neuron_id: the neuron id to find the bit in the bitfield
        :return: the bit
        """
        word_id = int(math.floor(neuron_id // self._BITS_IN_A_WORD))
        bit_in_word = neuron_id % self._BITS_IN_A_WORD
        flag = (bit_field[word_id] >> bit_in_word) & self._BIT_MASK
        return flag

    def _read_in_bit_fields(
            self, transceiver, chip_x, chip_y, bit_field_chip_base_addresses):
        """ reads in the bitfields from the cores
        
        :param transceiver: SpiNNMan instance
        :param chip_x: chip x coord
        :param chip_y: chip y coord
        :param bit_field_chip_base_addresses: dict of core id to base address  
        :return: dict of lists of processor id to bitfields. 
        """

        # data holder
        bit_fields_by_processor = defaultdict(list)
        bit_field_by_key = DefaultOrderedDict(list)

        # read in for each app vertex that would have a bitfield
        for processor_id in bit_field_chip_base_addresses.keys():
            bit_field_base_address = bit_field_chip_base_addresses[processor_id]

            # read how many bitfields there are
            n_bit_field_entries = struct.unpack("<I", transceiver.read_memory(
                chip_x, chip_y, bit_field_base_address,
                self._BYTES_PER_WORD))[0]
            reading_address = bit_field_base_address + self._BYTES_PER_WORD

            # read in each bitfield
            for bit_field_index in range(0, n_bit_field_entries):

                # master pop key
                master_pop_key = struct.unpack("<I", transceiver.read_memory(
                    chip_x, chip_y, reading_address, self._BYTES_PER_WORD))[0]
                reading_address += self._BYTES_PER_WORD

                # how many words the bitfield uses
                n_words_to_read = struct.unpack("<I", transceiver.read_memory(
                    chip_x, chip_y, reading_address, self._BYTES_PER_WORD))[0]
                reading_address += self._BYTES_PER_WORD

                # get bitfield words
                bit_field = struct.unpack(
                    "<{}I".format(n_words_to_read),
                    transceiver.read_memory(
                        chip_x, chip_y, reading_address,
                        n_words_to_read * constants.WORD_TO_BYTE_MULTIPLIER))
                reading_address += (
                    n_words_to_read * constants.WORD_TO_BYTE_MULTIPLIER)

                # add to the bitfields tracker
                bit_fields_by_processor[processor_id].append(
                    (master_pop_key, bit_field))
                bit_field_by_key[master_pop_key].append(
                    (bit_field, processor_id))

        return bit_fields_by_processor, bit_field_by_key

    def _start_binary_search(
            self, router_table, bit_fields_by_key, target_length):
        """ start binary search of the merging of bitfield to router table
        
        :param router_table: uncompressed router table 
        :param bit_fields_by_key: the sorted bitfields
        :param target_length: length to compress to
        :return: final_routing_table, bit_fields_merged
        """

        # try first just uncompressed. so see if its possible
        try:
            self._last_successful = self.do_mundy_host_compression(
                [router_table], target_length, router_table.x, router_table.y)
            self._last_successful_bit_fields_merged = []
        except MinimisationFailedError:
            raise PacmanAlgorithmFailedToCompleteException(
                algorithm="host bitfield router compressor", tb=None,
                exception="cant compress the uncompressed routing tables, "
                          "regardless of bitfield merging. System is "
                          "fundamentally flawed here")

        max_size = 0
        for master_pop_key in bit_fields_by_key.keys():
            max_size += len(bit_fields_by_key[master_pop_key])

        find_max_success(max_size, functools.partial(
            self._binary_search_check, bit_fields_by_key=bit_fields_by_key,
            routing_table=router_table, target_length=target_length))

    def _binary_search_check(
            self, mid_point, bit_fields_by_key, routing_table, target_length):
        """ check function for fix max success
        
        :param mid_point: the point if the list to stop at
        :return: bool that is true if it compresses 
        """

        # find new set of bitfields to try from midpoint
        new_set_of_bit_fields = DefaultOrderedDict(list)

        values_added = 0
        for master_pop_key in bit_fields_by_key.keys():
            n_processor_bit_fields = len(bit_fields_by_key[master_pop_key])
            if values_added + n_processor_bit_fields <= mid_point:
                new_set_of_bit_fields[master_pop_key].extend(
                    bit_fields_by_key[master_pop_key])
                values_added += n_processor_bit_fields
            else:
                values_to_add = mid_point - values_added
                if values_to_add != 0:
                    new_set_of_bit_fields[master_pop_key].extend(
                        bit_fields_by_key[master_pop_key][0:values_to_add])
                values_added += values_to_add

        # convert bitfields into router tables
        bit_field_router_tables = self._convert_bitfields_into_router_tables(
            routing_table, new_set_of_bit_fields)

        # try to compress
        try:
            self._last_successful = self.do_mundy_host_compression(
                bit_field_router_tables, target_length, routing_table.x,
                routing_table.y)
            self._last_successful_bit_fields_merged = new_set_of_bit_fields
            return True
        except MinimisationFailedError:
            return False

    def do_mundy_host_compression(
            self, router_tables, target_length, chip_x, chip_y):
        """ attempts to covert the mega router tables into 1 router table. will\
        raise a MinimisationFailedError exception if it fails to compress to \
        the correct length
        
        :param router_tables: the set of router tables that together need to be\
        merged into 1 router table
        :param target_length: the number 
        :param chip_x: 
        :param chip_y: 
        :return: 
        :throws: MinimisationFailedError 
        """

        # convert to rig format
        entries = list()
        for router_table in router_tables:
            entries.extend(MundyRouterCompressor.convert_to_mundy_format(
                router_table))

        # compress the router entries
        compressed_router_table_entries = \
            rigs_compressor.minimise(entries, target_length)

        # convert back to pacman model
        compressed_pacman_table = \
            MundyRouterCompressor.convert_to_pacman_router_table(
                compressed_router_table_entries, chip_x, chip_y,
                self.MAX_SUPPORTED_LENGTH)

        return compressed_pacman_table

    def _remove_merged_bitfields_from_cores(
            self, bit_fields_merged, chip_x, chip_y, transceiver,
            bit_field_chip_base_addresses, bit_fields_by_processor):
        """ goes to sdram and removes said merged entries from the cores \
        bitfield region

        :param bit_fields_merged: the bitfields that were merged into router \
        table
        :param chip_x: the chip x coord from which this happened
        :param chip_y: the chip y coord from which this happened
        :param transceiver: spinnman instance
        :param bit_field_chip_base_addresses: base addresses of chip bit fields
        :param bit_fields_by_processor: map of processor to bitfields
        :rtype: None 
        """

        # get data back ina  form useful for write back
        merged_bit_field_by_core = defaultdict(list)
        for master_pop_key in bit_fields_merged:
            for (_, processor_id) in bit_fields_merged[master_pop_key]:
                merged_bit_field_by_core[processor_id].append(master_pop_key)

        # process the separate cores
        for processor_id in bit_fields_by_processor.keys():

            # amount of entries to remove
            new_total = (
                len(bit_fields_by_processor[processor_id]) -
                len(merged_bit_field_by_core[processor_id]))

            # base address for the region
            bit_field_base_address = bit_field_chip_base_addresses[processor_id]
            writing_address = bit_field_base_address

            # write correct number of elements.
            transceiver.write_memory(
                chip_x, chip_y, writing_address, self._ONE_WORDS.pack(
                    new_total), self._BYTES_PER_WORD)
            writing_address += self._BYTES_PER_WORD

            # iterate through the original bitfields and omit the ones deleted
            for (master_pop_key, bit_field) in bit_fields_by_processor[
                    processor_id]:
                if master_pop_key not in merged_bit_field_by_core[processor_id]:

                    # write key and n words
                    transceiver.write_memory(
                        chip_x, chip_y, writing_address,
                        self._TWO_WORDS.pack(master_pop_key, len(bit_field)),
                        self._BYTES_PER_WORD * 2)
                    writing_address += self._BYTES_PER_WORD * 2

                    # write bitfield words
                    data = struct.pack(
                        "<{}I".format(len(bit_field)), *bit_field)
                    transceiver.write_memory(
                        chip_x, chip_y, writing_address, data,
                        len(bit_field) * self._BYTES_PER_WORD)
                    writing_address += len(bit_field) * self._BYTES_PER_WORD

    def _record_changes(
            self, router_table, final_routing_table, bit_fields_merged,
            report_out):

        n_bit_fields_merged = 0
        for master_pop_key in bit_fields_merged.keys():
            n_bit_fields_merged += len(bit_fields_merged[master_pop_key])

        report_out.write(
            "Table {}:{} has {} bitfields merged. These are \n\n".format(
                router_table.x, router_table.y, n_bit_fields_merged))
        for master_pop_key in bit_fields_merged.keys():
            for (_, processor_id) in bit_fields_merged[master_pop_key]:
                report_out.write("bitfield on core {} for key {} \n".format(
                    processor_id, master_pop_key))

        report_out.write("\n\n\n")

        report_out.write(
            "The final routing table contains {} entries and is below: "
            "\n\n".format(final_routing_table.number_of_entries))

        report_out.write(
            "{: <5s} {: <10s} {: <10s} {: <10s} {: <7s} {}\n".format(
                "Index", "Key", "Mask", "Route", "Default", "[Cores][Links]"))
        report_out.write(
            "{:-<5s} {:-<10s} {:-<10s} {:-<10s} {:-<7s} {:-<14s}\n".format(
                "", "", "", "", "", ""))
        line_format = "{: >5d} {}\n"

        entry_count = 0
        n_defaultable = 0
        for entry in final_routing_table.multicast_routing_entries:
            index = entry_count & self._LOWER_16_BITS
            entry_str = line_format.format(index, format_route(entry))
            entry_count += 1
            if entry.defaultable:
                n_defaultable += 1
            report_out.write(entry_str)
        report_out.write("{} Defaultable entries\n".format(n_defaultable))
