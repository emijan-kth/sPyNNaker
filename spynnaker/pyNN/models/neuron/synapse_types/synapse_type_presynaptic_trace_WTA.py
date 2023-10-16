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
from .synapse_type_presynaptic_trace import SynapseTypePresynapticTrace


class SynapseTypePresynapticTraceWTA(SynapseTypePresynapticTrace):
    # """
    # This represents a synapse type with two delta synapses,
    # a presynaptic trace and a WTA reset synapse.
    # """

    @overrides(SynapseTypePresynapticTrace.get_n_synapse_types)
    def get_n_synapse_types(self):
        return 4

    @overrides(SynapseTypePresynapticTrace.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        if target == "excitatory":
            return 0
        elif target == "inhibitory":
            return 1
        elif target == "presynaptic-trace":
            return 2
        elif target == "WTA-reset":
            return 3
        return None

    @overrides(SynapseTypePresynapticTrace.get_synapse_targets)
    def get_synapse_targets(self):
        return "excitatory", "inhibitory", "presynaptic-trace", "WTA-reset"
