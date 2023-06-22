# Copyright (c) 2017 The University of Manchester
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

from spinn_utilities.overrides import overrides
from spinn_front_end_common.interface.ds import DataType
from .abstract_synapse_type import AbstractSynapseType
from spynnaker.pyNN.utilities.struct import Struct
from spynnaker.pyNN.data import SpynnakerDataView

ISYN_EXC = "isyn_exc"
ISYN_INH = "isyn_inh"
TAU_SYN_TRACE = 'tau_syn_trace'
ALPHA = 'alpha'
ISYN_TRACE = "isyn_trace"
TIMESTEP_MS = "timestep_ms"



class SynapseTypePresynapticTrace(AbstractSynapseType):
    """
    This represents a synapse type with two delta synapses and
    a presynaptic trace.
    """
    __slots__ = [
        "__isyn_exc",
        "__isyn_inh",
        "__tau_syn_trace",
        "__alpha",
        "__isyn_trace",
        ]

    def __init__(self, isyn_exc, isyn_inh, tau_syn_trace, alpha, isyn_trace):
        """
        :param isyn_exc: :math:`I^{syn}_e`
        :type isyn_exc: float or iterable(float) or
            ~spynnaker.pyNN.RandomDistribution or (mapping) function
        :param isyn_inh: :math:`I^{syn}_i`
        :type isyn_inh: float or iterable(float) or
            ~spynnaker.pyNN.RandomDistribution or (mapping) function
        :param tau_syn_trace: :math:`\tau^{syn}_{trace}`
        :type tau_syn_trace: float or iterable(float) or
            ~spynnaker.pyNN.RandomDistribution or (mapping) function
        :param alpha: :math:`\alpha`
        :type alpha: float or iterable(float) or
            ~spynnaker.pyNN.RandomDistribution or (mapping) function
        :param isyn_trace: :math:`I^{syn}_{trace}`
        :type isyn_trace: float or iterable(float) or
            ~spynnaker.pyNN.RandomDistribution or (mapping) function
        """
        super().__init__(
            [Struct([
                (DataType.S1615, ISYN_EXC),  # isyn_exc
                (DataType.S1615, ISYN_INH),  # isyn_inh
                (DataType.S1615, TAU_SYN_TRACE),
                (DataType.S1615, ISYN_TRACE),
                (DataType.S1615, ALPHA),
                (DataType.S1615, TIMESTEP_MS),
                ])],  
            {ISYN_EXC: "", ISYN_EXC: "", TAU_SYN_TRACE: 'mV', ALPHA: "", ISYN_TRACE: ""})
        self.__isyn_exc = isyn_exc
        self.__isyn_inh = isyn_inh
        self.__tau_syn_trace = tau_syn_trace
        self.__alpha = alpha
        self.__isyn_trace = isyn_trace

    @overrides(AbstractSynapseType.add_parameters)
    def add_parameters(self, parameters):
        parameters[TAU_SYN_TRACE] = self.__tau_syn_trace
        parameters[ALPHA] = self.__alpha
        parameters[TIMESTEP_MS] = (
            SpynnakerDataView.get_simulation_time_step_ms())

    @overrides(AbstractSynapseType.add_state_variables)
    def add_state_variables(self, state_variables):
        state_variables[ISYN_EXC] = self.__isyn_exc
        state_variables[ISYN_INH] = self.__isyn_inh
        state_variables[ISYN_TRACE] = self.__isyn_trace

    @overrides(AbstractSynapseType.get_n_synapse_types)
    def get_n_synapse_types(self):
        return 3

    @overrides(AbstractSynapseType.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        if target == "excitatory":
            return 0
        elif target == "inhibitory":
            return 1
        elif target == "presynaptic-trace":
            return 2
        return None

    @overrides(AbstractSynapseType.get_synapse_targets)
    def get_synapse_targets(self):
        return "excitatory", "inhibitory", "presynaptic-trace"

    @property
    def isyn_exc(self):
        return self.__isyn_exc

    @property
    def isyn_inh(self):
        return self.__isyn_inh

    @property
    def tau_syn_trace(self):
        return self.__tau_syn_trace

    @property
    def alpha(self):
        return self.__alpha

    @property
    def isyn_trace(self):
        return self.__isyn_trace
