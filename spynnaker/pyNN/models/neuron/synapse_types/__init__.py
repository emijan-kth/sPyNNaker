# Copyright (c) 2015 The University of Manchester
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

from .abstract_synapse_type import AbstractSynapseType
from .synapse_type_dual_exponential import SynapseTypeDualExponential
from .synapse_type_exponential import SynapseTypeExponential
from .synapse_type_delta import SynapseTypeDelta
from .synapse_type_presynaptic_trace import SynapseTypePresynapticTrace
from .synapse_type_presynaptic_trace_WTA import SynapseTypePresynapticTraceWTA
from .synapse_type_alpha import SynapseTypeAlpha
from .synapse_type_semd import SynapseTypeSEMD

__all__ = ["AbstractSynapseType", "SynapseTypeDualExponential",
           "SynapseTypeExponential", "SynapseTypeDelta",
           "SynapseTypePresynapticTrace", "SynapseTypePresynapticTraceWTA",
           "SynapseTypeAlpha", "SynapseTypeSEMD"]
