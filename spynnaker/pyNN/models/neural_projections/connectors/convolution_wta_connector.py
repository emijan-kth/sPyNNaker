import numpy
from spinn_utilities.overrides import overrides
from spinn_front_end_common.utilities.constants import BYTES_PER_WORD
from spynnaker.pyNN.exceptions import SynapticConfigurationException
from . import convolution_connector
import spynnaker.pyNN.models.neuron.local_only.local_only_convolution_WTA as local_only_convolution_WTA
from spynnaker.pyNN.utilities.constants import SPIKE_PARTITION_ID


CONNECTOR_CONFIG_SIZE = convolution_connector.CONNECTOR_CONFIG_SIZE + BYTES_PER_WORD

class ConvolutionWTAConnector(convolution_connector.ConvolutionConnector):

    @property
    def WTA_reset_receptor_type(self):
        return "WTA-reset"

    @overrides(convolution_connector.ConvolutionConnector.validate_connection)
    def validate_connection(self, application_edge, synapse_info):
        if not isinstance(synapse_info.synapse_dynamics, local_only_convolution_WTA.LocalOnlyConvolutionWTA):
            raise SynapticConfigurationException(
                "This connector must have a synapse_type of"
                " ConvolutionWTA")
        super().validate_connection(application_edge, synapse_info)

    @property
    @overrides(convolution_connector.ConvolutionConnector.parameters_n_bytes)
    def parameters_n_bytes(self):   
        return CONNECTOR_CONFIG_SIZE

    @overrides(convolution_connector.ConvolutionConnector.get_local_only_data)
    def get_local_only_data(
            self, app_edge, local_delay, delay_stage, weight_index):
        data = super().get_local_only_data(app_edge, local_delay, delay_stage, weight_index)

        WTA_reset_synapse_type = app_edge.post_vertex.get_synapse_id_by_target(
            self.WTA_reset_receptor_type)

        # Produce the values needed
        short_values = numpy.array([
            WTA_reset_synapse_type, 0], dtype="uint16")
        
        return numpy.concatenate((data, short_values.view("uint32")))


class ConvolutionWTAResetConnector(ConvolutionWTAConnector):

    def __init__(self, num_channels):
        super().__init__(((0,),))

        self.num_channels = num_channels

    @overrides(convolution_connector.ConvolutionConnector.validate_connection)
    def validate_connection(self, application_edge, synapse_info):
        if not isinstance(synapse_info.synapse_dynamics, local_only_convolution_WTA.LocalOnlyConvolutionWTA):
            raise SynapticConfigurationException(
                "This connector must have a synapse_type of"
                " ConvolutionWTA")

    @overrides(convolution_connector.ConvolutionConnector.get_connected_vertices)
    def get_connected_vertices(self, s_info, source_vertex, target_vertex):
        pre_vertices = numpy.array(
            source_vertex.splitter.get_out_going_vertices(SPIKE_PARTITION_ID))

        pre_slices = [m_vertex.vertex_slice for m_vertex in pre_vertices]

        pre_slices_x = [vtx_slice.get_slice(0) for vtx_slice in pre_slices]
        pre_slices_y = [vtx_slice.get_slice(1) for vtx_slice in pre_slices]

        pre_ranges_in_post = [[[py.start, px.start], [py.stop - 1, px.stop - 1]]
                      for px, py in zip(pre_slices_x, pre_slices_y)]

        pre_ranges_in_post = numpy.array(pre_ranges_in_post)

        pre_vertex_in_post_layer_upper_left = pre_ranges_in_post[:,0]
        pre_vertex_in_post_layer_lower_right = pre_ranges_in_post[:,1]

        post_vertices = target_vertex.splitter.get_in_coming_vertices(
                SPIKE_PARTITION_ID)

        connected = list()

        for post in post_vertices:
            post_slice = post.vertex_slice
            post_slice_x = post_slice.get_slice(0)
            post_slice_y = post_slice.get_slice(1)

            # Get ranges allowed in post vertex
            min_x = post_slice_x.start
            max_x = post_slice_x.stop - 1
            min_y = post_slice_y.start
            max_y = post_slice_y.stop - 1

            # Test that the start coords are in range i.e. less than max
            start_in_range = numpy.logical_not(
                numpy.any(pre_vertex_in_post_layer_upper_left > [max_y, max_x], axis=1))
            # Test that the end coords are in range i.e. more than min
            end_in_range = numpy.logical_not(
                numpy.any(pre_vertex_in_post_layer_lower_right < [min_y, min_x], axis=1))
            # When both things are true, we have a vertex in range
            pre_in_range = pre_vertices[
                numpy.logical_and(start_in_range, end_in_range)]
            connected.append((post, pre_in_range))

        # print("\n*******\nResult of get_connected_vertices:")
        # for i, (post, pre) in enumerate(connected):
        #     print(f"{i}: ({pre}) -> ({post})")
        # print("******\n")

        return connected
