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

import pyNN.spiNNaker as p
from spynnaker.pyNN.utilities import neo_convertor
from spinnaker_testbase import BaseTestCase


def do_run():
    p.setup(timestep=1, min_delay=1)

    spiker = p.Population(1, p.SpikeSourceArray(spike_times=[[5, 25]]),
                          label='inputSSA')

    if_pop = p.Population(1, p.IF_cond_exp(), label='pop')

    if_pop.record("spikes")
    if_pop.record("v")

    runtime = 30

    # Create projection with delay such that the second spike occurs after
    # the run has finished
    weight = 5.0
    delay = 7
    p.Projection(spiker, if_pop, p.OneToOneConnector(),
                 synapse_type=p.StaticSynapse(weight=weight, delay=delay),
                 receptor_type="excitatory", source=None, space=None)

    p.run(runtime)
    all1 = if_pop.get_data(["spikes", "v"])

    # Reset (to time=0) and run again
    p.reset()
    p.run(runtime)
    all2 = if_pop.get_data(["spikes", "v"])

    p.end()

    return (all1, all2)


class ResetClearsDelayedSpikeTest(BaseTestCase):
    def check_run(self):
        all1, all2 = do_run()
        spikes1 = neo_convertor.convert_spiketrains(
            all1.segments[0].spiketrains)
        spikes2 = neo_convertor.convert_spiketrains(
            all2.segments[1].spiketrains)
        self.assertEqual(spikes1.all(), spikes2.all())
        v1 = neo_convertor.convert_data(all1, name="v", run=0)
        v2 = neo_convertor.convert_data(all2, name="v", run=1)
        self.assertEqual(v1.all(), v2.all())

    def test_run(self):
        self.runsafe(self.check_run)


if __name__ == '__main__':
    all1, all2 = do_run()
    spikes1 = neo_convertor.convert_spiketrains(all1.segments[0].spiketrains)
    print(spikes1)
    spikes2 = neo_convertor.convert_spiketrains(all2.segments[1].spiketrains)
    print(spikes2)
    v1 = neo_convertor.convert_data(all1, name="v", run=0)
    print(v1)
    v2 = neo_convertor.convert_data(all2, name="v", run=1)
    print(v2)
