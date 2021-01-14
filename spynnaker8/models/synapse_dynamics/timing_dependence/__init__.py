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

"""
.. warning::
    Using classes via this module is deprecated. Please use
    :py:mod:`spynnaker.pyNN.models.neuron.plasticity.stdp.timing_dependence`
    instead.
"""

from .spike_pair import TimingDependenceSpikePair
from .pfister_spike_triplet import TimingDependencePfisterSpikeTriplet
from .recurrent import TimingDependenceRecurrent
from .spike_nearest_pair import TimingDependenceSpikeNearestPair
from .vogels_2011 import TimingDependenceVogels2011

__all__ = ["TimingDependenceSpikePair",
           "TimingDependencePfisterSpikeTriplet", "TimingDependenceRecurrent",
           "TimingDependenceSpikeNearestPair", "TimingDependenceVogels2011"]
