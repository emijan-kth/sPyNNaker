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

from .abstract_pynn_neuron_model import AbstractPyNNNeuronModel
from spynnaker.pyNN.models.neuron.implementations.abstract_neuron_impl import AbstractNeuronImpl
from spinn_utilities.overrides import overrides

_population_parameters = dict(
    AbstractPyNNNeuronModel.default_population_parameters)
_population_parameters["n_steps_per_timestep"] = 1


class WTA(AbstractPyNNNeuronModel):
    """
    Winner-takes-it-all stage two.
    """

    __slots__ = []

    default_population_parameters = _population_parameters

    def __init__(
            self):
        """
        """
        super().__init__(WTAImpl(
            model_name="WTA",
            binary="WTA"))


class WTAImpl(AbstractNeuronImpl):
    """
    WTA neuron implementation.
    """

    __slots__ = [
        "__model_name",
        "__binary"
    ]

    _RECORDABLES = []

    _RECORDABLE_DATA_TYPES = {
    }

    _RECORDABLE_UNITS = {
    }

    def __init__(
            self, model_name, binary):
        """
        :param str model_name:
        :param str binary:
        """
        self.__model_name = model_name
        self.__binary = binary

    @property
    @overrides(AbstractNeuronImpl.model_name)
    def model_name(self):
        return self.__model_name

    @property
    @overrides(AbstractNeuronImpl.binary_name)
    def binary_name(self):
        return self.__binary

    @property
    @overrides(AbstractNeuronImpl.structs)
    def structs(self):
        structs = []
        return structs

    @overrides(AbstractNeuronImpl.get_global_weight_scale)
    def get_global_weight_scale(self):
        return 1.0

    @overrides(AbstractNeuronImpl.get_n_synapse_types)
    def get_n_synapse_types(self):
        return 1

    @overrides(AbstractNeuronImpl.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    @overrides(AbstractNeuronImpl.get_synapse_targets)
    def get_synapse_targets(self):
        return "excitatory"

    @overrides(AbstractNeuronImpl.get_recordable_variables)
    def get_recordable_variables(self):
        return self._RECORDABLES

    @overrides(AbstractNeuronImpl.get_recordable_units)
    def get_recordable_units(self, variable):
        return self._RECORDABLE_UNITS[variable]

    @overrides(AbstractNeuronImpl.get_recordable_data_types)
    def get_recordable_data_types(self):
        return self._RECORDABLE_DATA_TYPES

    @overrides(AbstractNeuronImpl.is_recordable)
    def is_recordable(self, variable):
        return variable in self._RECORDABLES

    @overrides(AbstractNeuronImpl.get_recordable_variable_index)
    def get_recordable_variable_index(self, variable):
        return self._RECORDABLES.index(variable)

    @overrides(AbstractNeuronImpl.add_parameters)
    def add_parameters(self, parameters):
        pass

    @overrides(AbstractNeuronImpl.add_state_variables)
    def add_state_variables(self, state_variables):
        pass

    @overrides(AbstractNeuronImpl.get_units)
    def get_units(self, variable):
        raise KeyError(
            f"The parameter {variable} does not exist")

    @property
    @overrides(AbstractNeuronImpl.is_conductance_based)
    def is_conductance_based(self):
        return False

    def __getitem__(self, key):
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute {key}")
