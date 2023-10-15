# Copyright (c) 2021 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy
from math import ceil
from spinn_utilities.overrides import overrides
from spinn_front_end_common.interface.ds import DataType
from spinn_front_end_common.utilities.constants import (
    BYTES_PER_SHORT, BYTES_PER_WORD)
from spynnaker.pyNN.exceptions import SynapticConfigurationException
import spynnaker.pyNN.models.neural_projections.connectors.convolution_wta_connector as convolution_wta_connector
from spynnaker.pyNN.models.common.local_only_2d_common import (
    get_div_const, get_rinfo_for_source, get_sources_for_target,
    BITS_PER_SHORT, N_COLOUR_BITS_BITS, KEY_INFO_SIZE,
    get_first_and_last_slice)
from . import local_only_convolution


#: Size of convolution config main bytes
CONV_CONFIG_SIZE = local_only_convolution.CONV_CONFIG_SIZE

#: Size of source information
SOURCE_INFO_SIZE = local_only_convolution.SOURCE_INFO_SIZE + BYTES_PER_WORD


class LocalOnlyConvolutionWTA(local_only_convolution.LocalOnlyConvolution):

    def __init__(self, delay=None):
        """
        :param float delay:
            The delay used in the connection; by default 1 time step
        """
        super().__init__(delay)

    @overrides(local_only_convolution.LocalOnlyConvolution.merge)
    def merge(self, synapse_dynamics):  
        if not type(synapse_dynamics) is LocalOnlyConvolutionWTA:
            raise SynapticConfigurationException(
                "All targets of this Population must have a synapse_type of"
                " ConvolutionWTA")
        return synapse_dynamics

    @overrides(local_only_convolution.LocalOnlyConvolution.get_parameters_usage_in_bytes)
    def get_parameters_usage_in_bytes(
            self, n_atoms, incoming_projections):
        n_bytes = 0
        kernel_bytes = 0
        connectors_seen = set()
        edges_seen = set()
        for incoming in incoming_projections:
            # pylint: disable=protected-access
            s_info = incoming._synapse_information
            if not isinstance(s_info.connector, convolution_wta_connector.ConvolutionWTAConnector):
                raise SynapticConfigurationException(
                    "Only ConvolutionWTAConnector can be used with a synapse type"
                    " of ConvolutionWTA")
            # pylint: disable=protected-access
            app_edge = incoming._projection_edge
            if app_edge not in edges_seen:
                edges_seen.add(app_edge)
                n_bytes += SOURCE_INFO_SIZE
            if s_info.connector not in connectors_seen:
                connectors_seen.add(s_info.connector)
                kernel_bytes += s_info.connector.kernel_n_bytes
            n_bytes += s_info.connector.parameters_n_bytes

        if kernel_bytes % BYTES_PER_WORD != 0:
            kernel_bytes += BYTES_PER_SHORT

        return CONV_CONFIG_SIZE + n_bytes + kernel_bytes

    @overrides(local_only_convolution.LocalOnlyConvolution.write_parameters)
    def write_parameters(self, spec, region, machine_vertex, weight_scales):

        # Get incoming sources for this vertex
        app_vertex = machine_vertex.app_vertex
        sources = self._LocalOnlyConvolution__get_sources_for_target(app_vertex)

        size = self.get_parameters_usage_in_bytes(
            machine_vertex.vertex_slice, app_vertex.incoming_projections)
        spec.reserve_memory_region(region, size, label="LocalOnlyConvolution")
        spec.switch_write_focus(region)

        # Get spec for each incoming source
        connector_weight_index = dict()
        next_weight_index = 0
        source_data = list()
        connector_data = list()
        weight_data = list()
        for pre_vertex, source_infos in sources.items():

            # Add connectors as needed
            first_conn_index = len(connector_data)

            is_WTA_reset = False
            
            for source in source_infos:
                # pylint: disable=protected-access
                conn = source.projection._synapse_information.connector

                if isinstance(conn, convolution_wta_connector.ConvolutionWTAResetConnector):
                    is_WTA_reset = True

                app_edge = source.projection._projection_edge

                # Work out whether the connector needs a new weight index
                if conn in connector_weight_index:
                    weight_index = connector_weight_index[conn]
                else:
                    weight_index = next_weight_index
                    connector_weight_index[conn] = weight_index
                    next_weight_index += conn.kernel_n_weights
                    weight_data.append(conn.get_encoded_kernel_weights(
                        app_edge, weight_scales))

                connector_data.append(conn.get_local_only_data(
                    app_edge, source.local_delay, source.delay_stage,
                    weight_index))

            # Get the source routing information
            r_info, core_mask, mask_shift = get_rinfo_for_source(
                pre_vertex)

            # Get the width / height per core / last_core
            first_slice, last_slice = get_first_and_last_slice(pre_vertex)
            width_per_core = first_slice.shape[0]
            height_per_core = first_slice.shape[1]
            width_on_last_core = last_slice.shape[0]
            height_on_last_core = last_slice.shape[1]

            # Get cores per width / height
            pre_shape = list(pre_vertex.atoms_shape)
            cores_per_width = int(ceil(pre_shape[0] / width_per_core))
            cores_per_height = int(ceil(pre_shape[1] / height_per_core))

            # Add the key and mask...
            source_data.extend([r_info.key, r_info.mask])
            # ... start connector index, n_colour_bits, count of connectors ...
            source_data.append(
                (len(source_infos) << BITS_PER_SHORT) +
                (pre_vertex.n_colour_bits <<
                 (BITS_PER_SHORT - N_COLOUR_BITS_BITS)) +
                first_conn_index)
            # ... core mask, mask shift ...
            source_data.append((mask_shift << BITS_PER_SHORT) + core_mask)
            # ... height / width per core ...
            source_data.append(
                (width_per_core << BITS_PER_SHORT) + height_per_core)
            # ... height / width last core ...
            source_data.append(
                (width_on_last_core << BITS_PER_SHORT) + height_on_last_core)
            # ... cores per height / width ...
            source_data.append(
                (cores_per_width << BITS_PER_SHORT) + cores_per_height)
            # ... 1 / width per core ...
            source_data.append(get_div_const(width_per_core))
            # ... 1 / width last core ...
            source_data.append(get_div_const(width_on_last_core))
            # ... 1 / cores_per_width
            source_data.append(get_div_const(cores_per_width))

            source_data.append(is_WTA_reset)

        if next_weight_index % 2 != 0:
            weight_data.append(numpy.array([0], dtype="int16"))

        # Write the common spec
        post_slice = machine_vertex.vertex_slice
        post_start = numpy.array(post_slice.start)
        post_shape = numpy.array(post_slice.shape)
        post_end = (post_start + post_shape) - 1
        spec.write_value(post_start[1], data_type=DataType.INT16)
        spec.write_value(post_start[0], data_type=DataType.INT16)
        spec.write_value(post_end[1], data_type=DataType.INT16)
        spec.write_value(post_end[0], data_type=DataType.INT16)
        spec.write_value(post_shape[1], data_type=DataType.INT16)
        spec.write_value(post_shape[0], data_type=DataType.INT16)
        spec.write_value(len(sources))
        spec.write_value(len(connector_data))
        spec.write_value(next_weight_index)

        # Write the data
        # pylint: disable=unexpected-keyword-arg
        spec.write_array(numpy.array(source_data, dtype="uint32"))
        spec.write_array(numpy.concatenate(connector_data, dtype="uint32"))
        spec.write_array(
            numpy.concatenate(weight_data, dtype="int16").view("uint32"))

    @overrides(local_only_convolution.LocalOnlyConvolution.get_auxiliary_synapse_indices)
    def get_auxiliary_synapse_indices(self, incoming_projection):
        # pylint: disable=protected-access
        post = incoming_projection._projection_edge.post_vertex
        conn = incoming_projection._synapse_information.connector
        
        if not isinstance(conn, convolution_wta_connector.ConvolutionWTAConnector):
            raise SynapticConfigurationException(
                "Only ConvolutionWTAConnector can be used with a synapse type"
                " of ConvolutionWTA")

        super_indices = super().get_auxiliary_synapse_indices(incoming_projection)
        return (() if super_indices is None else super_indices) + \
            (post.get_synapse_id_by_target(conn.WTA_reset_receptor_type),)
