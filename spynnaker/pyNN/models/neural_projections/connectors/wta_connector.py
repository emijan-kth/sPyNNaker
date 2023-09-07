from spinn_utilities.overrides import overrides
from .convolution_connector import ConvolutionConnector

class WTAConnector(ConvolutionConnector):

    def __init__(self):
        super(WTAConnector, self).__init__(((0,),))

    @overrides(ConvolutionConnector.validate_connection)
    def validate_connection(self, application_edge, synapse_info):
        pass
