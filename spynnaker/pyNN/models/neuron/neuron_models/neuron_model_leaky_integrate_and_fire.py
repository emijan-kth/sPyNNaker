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

import numpy
from spinn_utilities.overrides import overrides
from data_specification.enums import DataType
from spynnaker.pyNN.models.neuron.implementations import (
    AbstractStandardNeuronComponent)
from spynnaker.pyNN.utilities.struct import Struct

V = "v"
V_REST = "v_rest"
TAU_M = "tau_m"
CM = "cm"
I_OFFSET = "i_offset"
V_RESET = "v_reset"
TAU_REFRAC = "tau_refrac"
COUNT_REFRAC = "count_refrac"
R_MEMBRANE = "r_membrane"
EXP_TC = "exp_tc"
INV_TAU_REFRAC = "inv_tau_refrac"


class NeuronModelLeakyIntegrateAndFire(AbstractStandardNeuronComponent):
    """ Classic leaky integrate and fire neuron model.
    """
    __slots__ = [
        "__v_init",
        "__v_rest",
        "__tau_m",
        "__cm",
        "__i_offset",
        "__v_reset",
        "__tau_refrac"]

    def __init__(
            self, v_init, v_rest, tau_m, cm, i_offset, v_reset, tau_refrac):
        r"""
        :param v_init: :math:`V_{init}`
        :type v_init:
            float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        :param v_rest: :math:`V_{rest}`
        :type v_rest:
            float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        :param tau_m: :math:`\tau_{m}`
        :type tau_m:
            float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        :param cm: :math:`C_m`
        :type cm: float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        :param i_offset: :math:`I_{offset}`
        :type i_offset:
            float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        :param v_reset: :math:`V_{reset}`
        :type v_reset:
            float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        :param tau_refrac: :math:`\tau_{refrac}`
        :type tau_refrac:
            float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        """
        super().__init__(
            [Struct([
                (DataType.S1615, V),
                (DataType.S1615, V_REST),
                (DataType.S1615, R_MEMBRANE),
                (DataType.S1615, EXP_TC),
                (DataType.S1615, I_OFFSET),
                (DataType.INT32, COUNT_REFRAC),
                (DataType.S1615, V_RESET),
                (DataType.INT32, INV_TAU_REFRAC)])],
            {V: 'mV', V_REST: 'mV', TAU_M: 'ms', CM: 'nF', I_OFFSET: 'nA',
             V_RESET: 'mV', TAU_REFRAC: 'ms'})

        if v_init is None:
            v_init = v_rest
        self.__v_init = v_init
        self.__v_rest = v_rest
        self.__tau_m = tau_m
        self.__cm = cm
        self.__i_offset = i_offset
        self.__v_reset = v_reset
        self.__tau_refrac = tau_refrac

    @overrides(AbstractStandardNeuronComponent.get_n_cpu_cycles)
    def get_n_cpu_cycles(self, n_neurons):
        # A bit of a guess
        return 100 * n_neurons

    @overrides(AbstractStandardNeuronComponent.add_parameters)
    def add_parameters(self, parameters):
        parameters[V_REST] = self.__v_rest
        parameters[TAU_M] = self.__tau_m
        parameters[CM] = self.__cm
        parameters[I_OFFSET] = self.__i_offset
        parameters[V_RESET] = self.__v_reset
        parameters[TAU_REFRAC] = self.__tau_refrac

    @overrides(AbstractStandardNeuronComponent.add_state_variables)
    def add_state_variables(self, state_variables):
        state_variables[V] = self.__v_init
        state_variables[COUNT_REFRAC] = 0

    @overrides(AbstractStandardNeuronComponent.get_precomputed_values)
    def get_precomputed_values(self, parameters, state_variables, ts):
        return {R_MEMBRANE: parameters[TAU_M] / parameters[CM],
                EXP_TC: parameters[TAU_M].apply_operation(
                    operation=lambda x: numpy.exp(float(-ts) / (1000.0 * x))),
                INV_TAU_REFRAC: parameters[TAU_REFRAC].apply_operation(
                    operation=lambda x: int(numpy.ceil(x / (ts / 1000.0))))}

    @property
    def v_init(self):
        """ Settable model parameter: :math:`V_{init}`

        :rtype: float
        """
        return self.__v_init

    @property
    def v_rest(self):
        """ Settable model parameter: :math:`V_{rest}`

        :rtype: float
        """
        return self.__v_rest

    @property
    def tau_m(self):
        r""" Settable model parameter: :math:`\tau_{m}`

        :rtype: float
        """
        return self.__tau_m

    @property
    def cm(self):
        """ Settable model parameter: :math:`C_m`

        :rtype: float
        """
        return self.__cm

    @property
    def i_offset(self):
        """ Settable model parameter: :math:`I_{offset}`

        :rtype: float
        """
        return self.__i_offset

    @property
    def v_reset(self):
        """ Settable model parameter: :math:`V_{reset}`

        :rtype: float
        """
        return self.__v_reset

    @property
    def tau_refrac(self):
        r""" Settable model parameter: :math:`\tau_{refrac}`

        :rtype: float
        """
        return self.__tau_refrac
