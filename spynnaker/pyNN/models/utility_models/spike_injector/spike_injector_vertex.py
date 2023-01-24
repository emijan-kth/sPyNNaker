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
from spinn_utilities.log import FormatAdapter
from spinn_utilities.overrides import overrides
from spinn_front_end_common.utility_models import ReverseIpTagMultiCastSource
from spynnaker.pyNN.data import SpynnakerDataView
from spynnaker.pyNN.models.common import EIEIOSpikeRecorder
from spynnaker.pyNN.utilities.buffer_data_type import BufferDataType
from spynnaker.pyNN.utilities.constants import SPIKE_PARTITION_ID
from spynnaker.pyNN.models.common import PopulationApplicationVertex

logger = FormatAdapter(logging.getLogger(__name__))


class SpikeInjectorVertex(
        ReverseIpTagMultiCastSource, PopulationApplicationVertex):
    """ An Injector of Spikes for PyNN populations.  This only allows the user\
        to specify the virtual_key of the population to identify the population
    """
    __slots__ = [
        "__spike_recorder"]

    default_parameters = {
        'label': "spikeInjector", 'port': None, 'virtual_key': None}

    SPIKE_RECORDING_REGION_ID = 0

    def __init__(
            self, n_neurons, label, port, virtual_key,
            reserve_reverse_ip_tag, splitter):
        # pylint: disable=too-many-arguments

        super().__init__(
            n_keys=n_neurons, label=label, receive_port=port,
            virtual_key=virtual_key,
            reserve_reverse_ip_tag=reserve_reverse_ip_tag,
            injection_partition_id=SPIKE_PARTITION_ID,
            splitter=splitter)

        # Set up for recording
        self.__spike_recorder = EIEIOSpikeRecorder()

    @overrides(PopulationApplicationVertex.get_recordable_variables)
    def get_recordable_variables(self):
        return ["spikes"]

    @overrides(PopulationApplicationVertex.set_recording)
    def set_recording(self, name, sampling_interval=None, indices=None):
        if name != "spikes":
            raise KeyError(f"Cannot record {name}")
        if sampling_interval is not None:
            logger.warning("Sampling interval currently not supported for "
                           "SpikeInjector so being ignored")
        if indices is not None:
            logger.warning("Indices currently not supported for "
                           "SpikeInjector so being ignored")
        self.enable_recording(True)
        self.__spike_recorder.record = True

    @overrides(PopulationApplicationVertex.get_recording_variables)
    def get_recording_variables(self):
        if self.__spike_recorder.record:
            return ["spikes"]
        return []

    @overrides(PopulationApplicationVertex.set_not_recording)
    def set_not_recording(self, name, indices=None):
        if name != "spikes":
            raise KeyError(f"Cannot record {name}")
        if indices is not None:
            logger.warning("Indices currently not supported for "
                           "SpikeSourceArray so being ignored")
        self.enable_recording(False)
        self.__spike_recorder.record = False

    @overrides(PopulationApplicationVertex.get_sampling_interval_ms)
    def get_sampling_interval_ms(self, name):
        if name != "spikes":
            raise KeyError(f"Cannot record {name}")
        return SpynnakerDataView.get_simulation_time_step_us()

    @overrides(PopulationApplicationVertex.get_data_type)
    def get_data_type(self, name):
        if name != "spikes":
            raise KeyError(f"Cannot record {name}")
        return None

    @overrides(PopulationApplicationVertex.get_buffer_data_type)
    def get_buffer_data_type(self, name):
        if name == "spikes":
            return BufferDataType.EIEIO_SPIKES
        raise KeyError(f"Cannot record {name}")

    @overrides(PopulationApplicationVertex.get_units)
    def get_units(self, name):
        if name == "spikes":
            return ""
        raise KeyError(f"Cannot record {name}")

    @overrides(PopulationApplicationVertex.get_recording_region)
    def get_recording_region(self, name):
        if name != "spikes":
            raise KeyError(f"Cannot record {name}")
        return 0

    @overrides(PopulationApplicationVertex.get_neurons_recording)
    def get_neurons_recording(self, name, vertex_slice):
        if name != "spikes":
            raise KeyError(f"Cannot record {name}")
        return vertex_slice.get_raster_ids(self.atoms_shape)

    def describe(self):
        """
        Returns a human-readable description of the cell or synapse type.

        The output may be customised by specifying a different template
        together with an associated template engine
        (see :py:mod:`pyNN.descriptions`).

        If template is None, then a dictionary containing the template context
        will be returned.
        """

        context = {
            "name": "SpikeInjector",
            "default_parameters": self.default_parameters,
            "default_initial_values": self.default_parameters,
            "parameters": {
                "port": self._receive_port,
                "virtual_key": self._virtual_key},
        }
        return context
