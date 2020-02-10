# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import numpy
from spinn_utilities.overrides import overrides
from spinn_front_end_common.utilities.constants import BYTES_PER_WORD
from .abstract_connector import AbstractConnector
from .abstract_generate_connector_on_machine import (
    AbstractGenerateConnectorOnMachine, ConnectorIDs)
from .abstract_connector_supports_views_on_machine import (
    AbstractConnectorSupportsViewsOnMachine)

logger = logging.getLogger(__file__)


class AllToAllConnector(AbstractGenerateConnectorOnMachine,
                        AbstractConnectorSupportsViewsOnMachine):
    """ Connects all cells in the presynaptic population to all cells in \
        the postsynaptic population.
    """

    __slots__ = [
        "__allow_self_connections"]

    def __init__(self, allow_self_connections=True, safe=True, callback=None,
                 verbose=None):
        """
        :param allow_self_connections:
            if the connector is used to connect a\
            Population to itself, this flag determines whether a neuron is\
            allowed to connect to itself, or only to other neurons in the\
            Population.
        :type allow_self_connections: bool
        """
        super(AllToAllConnector, self).__init__(safe, callback, verbose)
        self.__allow_self_connections = allow_self_connections

    def _connection_slices(self, pre_vertex_slice, post_vertex_slice,
                           synapse_info):
        """ Get a slice of the overall set of connections.
        """
        n_post_neurons = synapse_info.n_post_neurons
        stop_atom = post_vertex_slice.hi_atom + 1
        if (not self.__allow_self_connections and
                pre_vertex_slice is post_vertex_slice):
            n_post_neurons -= 1
            stop_atom -= 1
        return [
            slice(n + post_vertex_slice.lo_atom, n + stop_atom)
            for n in range(
                pre_vertex_slice.lo_atom * n_post_neurons,
                (pre_vertex_slice.hi_atom + 1) * n_post_neurons,
                n_post_neurons)]

    @overrides(AbstractConnector.get_delay_maximum)
    def get_delay_maximum(self, synapse_info):
        return self._get_delay_maximum(
            synapse_info.delays,
            synapse_info.n_pre_neurons * synapse_info.n_post_neurons)

    @overrides(AbstractConnector.get_n_connections_from_pre_vertex_maximum)
    def get_n_connections_from_pre_vertex_maximum(
            self, post_vertex_slice, synapse_info, min_delay=None,
            max_delay=None):
        # pylint: disable=too-many-arguments

        if min_delay is None or max_delay is None:
            return post_vertex_slice.n_atoms

        return self._get_n_connections_from_pre_vertex_with_delay_maximum(
            synapse_info.delays,
            synapse_info.n_pre_neurons * synapse_info.n_post_neurons,
            post_vertex_slice.n_atoms, min_delay, max_delay)

    @overrides(AbstractConnector.get_n_connections_to_post_vertex_maximum)
    def get_n_connections_to_post_vertex_maximum(self, synapse_info):
        return synapse_info.n_pre_neurons

    @overrides(AbstractConnector.get_weight_maximum)
    def get_weight_maximum(self, synapse_info):
        # pylint: disable=too-many-arguments
        n_conns = synapse_info.n_pre_neurons * synapse_info.n_post_neurons
        return self._get_weight_maximum(synapse_info.weights, n_conns)

    @overrides(AbstractConnector.create_synaptic_block)
    def create_synaptic_block(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            synapse_type, synapse_info):
        # pylint: disable=too-many-arguments
        n_connections = pre_vertex_slice.n_atoms * post_vertex_slice.n_atoms
        if (not self.__allow_self_connections and
                pre_vertex_slice is post_vertex_slice):
            n_connections -= post_vertex_slice.n_atoms
        connection_slices = self._connection_slices(
            pre_vertex_slice, post_vertex_slice, synapse_info)
        block = numpy.zeros(
            n_connections, dtype=AbstractConnector.NUMPY_SYNAPSES_DTYPE)

        if (not self.__allow_self_connections and
                pre_vertex_slice is post_vertex_slice):
            n_atoms = pre_vertex_slice.n_atoms
            block["source"] = numpy.where(numpy.diag(
                numpy.repeat(1, n_atoms)) == 0)[0]
            block["target"] = [block["source"][
                ((n_atoms * i) + (n_atoms - 1)) - j]
                for j in range(n_atoms) for i in range(n_atoms - 1)]
            block["source"] += pre_vertex_slice.lo_atom
            block["target"] += post_vertex_slice.lo_atom
        else:
            block["source"] = numpy.repeat(numpy.arange(
                pre_vertex_slice.lo_atom, pre_vertex_slice.hi_atom + 1),
                post_vertex_slice.n_atoms)
            block["target"] = numpy.tile(numpy.arange(
                post_vertex_slice.lo_atom, post_vertex_slice.hi_atom + 1),
                pre_vertex_slice.n_atoms)
        block["weight"] = self._generate_weights(
            n_connections, connection_slices, pre_vertex_slice,
            post_vertex_slice, synapse_info)
        block["delay"] = self._generate_delays(
            n_connections, connection_slices, pre_vertex_slice,
            post_vertex_slice, synapse_info)
        block["synapse_type"] = synapse_type
        return block

    def __repr__(self):
        return "AllToAllConnector()"

    @property
    def allow_self_connections(self):
        return self.__allow_self_connections

    @allow_self_connections.setter
    def allow_self_connections(self, new_value):
        self.__allow_self_connections = new_value

    @overrides(AbstractConnectorSupportsViewsOnMachine.get_view_lo_hi)
    def get_view_lo_hi(self, indexes):
        view_lo = indexes[0]
        view_hi = indexes[-1]
        return view_lo, view_hi

    @property
    @overrides(AbstractGenerateConnectorOnMachine.gen_connector_id)
    def gen_connector_id(self):
        return ConnectorIDs.ALL_TO_ALL_CONNECTOR.value

    @overrides(AbstractGenerateConnectorOnMachine.
               gen_connector_params)
    def gen_connector_params(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            synapse_type, synapse_info):
        params = []
        pre_view_lo = 0
        pre_view_hi = synapse_info.n_pre_neurons - 1
        if synapse_info.prepop_is_view:
            pre_view_lo, pre_view_hi = self.get_view_lo_hi(
                synapse_info.pre_population._indexes)

        params.extend([pre_view_lo, pre_view_hi])

        post_view_lo = 0
        post_view_hi = synapse_info.n_post_neurons - 1
        if synapse_info.postpop_is_view:
            post_view_lo, post_view_hi = self.get_view_lo_hi(
                synapse_info.post_population._indexes)

        params.extend([post_view_lo, post_view_hi])

        params.extend([self.allow_self_connections])

        return numpy.array(params, dtype="uint32")

    @property
    @overrides(AbstractGenerateConnectorOnMachine.
               gen_connector_params_size_in_bytes)
    def gen_connector_params_size_in_bytes(self):
        # view parameters + allow_self_connections
        return (4 + 1) * BYTES_PER_WORD
