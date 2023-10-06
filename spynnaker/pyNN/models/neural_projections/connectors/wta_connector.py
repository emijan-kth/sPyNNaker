import numpy
from spinn_utilities.overrides import overrides
from .convolution_connector import ConvolutionConnector
from spynnaker.pyNN.utilities.constants import SPIKE_PARTITION_ID

class WTAConnector(ConvolutionConnector):

    def __init__(self, num_channels):
        super(WTAConnector, self).__init__(((0,),))

        self.num_channels = num_channels

    @overrides(ConvolutionConnector.validate_connection)
    def validate_connection(self, application_edge, synapse_info):
        pass

    @overrides(ConvolutionConnector.get_connected_vertices)
    def get_connected_vertices(self, s_info, source_vertex, target_vertex):
        pre_vertices = numpy.array(
            source_vertex.splitter.get_out_going_vertices(SPIKE_PARTITION_ID))

        pre_slices = [m_vertex.vertex_slice for m_vertex in pre_vertices]

        pre_slices_x = [vtx_slice.get_slice(0) for vtx_slice in pre_slices]
        pre_slices_y = [vtx_slice.get_slice(1) for vtx_slice in pre_slices]

        pre_ranges_in_post = [[[py.start * self.num_channels, px.start], [(py.stop * self.num_channels) - 1, px.stop - 1]]
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

        return connected
