# Copyright (c) 2021 The University of Manchester
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

import numpy
import pyNN.spiNNaker as sim
from pyNN.space import Grid2D
import matplotlib.pyplot as plt
from spinnaker_testbase import BaseTestCase


def do_run():
    # 1x2 kernel, 1x1 stride
    run_testcase(
        in_shape=(2, 4),
        in_spike_times=(
            (0.0,), (), (), (1.0,),
            (), (2.0,), (3.0,), (),
        ),
        kernel=numpy.array(((2.0, 0.0),)),
        stride=numpy.array((1, 1), dtype='int32'),  # h, w
        expected_out_spike_times=(
            (), (), (2.0,),
            (3.0,), (4.0,), (),
        )
    )

    # 2x1 kernel, 1x1 stride
    run_testcase(
        in_shape=(2, 4),
        in_spike_times=(
            (0.0,), (), (), (1.0,),
            (), (2.0,), (3.0,), (),
        ),
        kernel=numpy.array(((2.0,), (0.0,),)),
        stride=numpy.array((1, 1), dtype='int32'),  # h, w
        expected_out_spike_times=(
            (), (3.0,), (4.0,), (),
        )
    )

    # 1x2 kernel, 1x2 stride version 1
    run_testcase(
        in_shape=(2, 4),
        in_spike_times=(
            (0.0,), (), (), (1.0,),
            (), (2.0,), (3.0,), (),
        ),
        kernel=numpy.array(((2.0, 0.0),)),
        stride=numpy.array((1, 2), dtype='int32'),  # h, w
        expected_out_spike_times=(
            (), (2.0,),
            (3.0,), (),
        )
    )

    # 1x2 kernel, 1x2 stride version 2
    run_testcase(
        in_shape=(2, 4),
        in_spike_times=(
            (0.0,), (), (), (1.0,),
            (), (2.0,), (3.0,), (),
        ),
        kernel=numpy.array(((0.0, 2.0),)),
        stride=numpy.array((1, 2), dtype='int32'),  # h, w
        expected_out_spike_times=(
            (1.0,), (),
            (), (4.0,),
        )
    )

    # 2x1 kernel, 2x1 stride
    run_testcase(
        in_shape=(4, 2),
        in_spike_times=(
            (0.0,), (),
            (), (1.0,),
            (), (2.0,),
            (3.0,), (),
        ),
        kernel=numpy.array(((2.0,), (0.0,),)),
        stride=numpy.array((2, 1), dtype='int32'),  # h, w
        expected_out_spike_times=(
            (), (2.0,),
            (4.0,), (),
        )
    )

    # 2x2 kernel, 1x1 stride
    run_testcase(
        in_shape=(2, 4),
        in_spike_times=(
            (0.0,), (), (), (1.0,),
            (), (2.0,), (3.0,), (),
        ),
        kernel=numpy.array((
            (2.0, 0.0,),
            (0.0, 2.0,),
        )),
        stride=numpy.array((1, 1), dtype='int32'),  # h, w
        expected_out_spike_times=(
            (1.0, 3.0,), (4.0,), (),
        )
    )

    # 2x2 kernel, 2x2 stride
    run_testcase(
        in_shape=(4, 4),
        in_spike_times=(
            (0.0,), (), (), (1.0,),
            (), (2.0,), (3.0,), (),
            (), (), (3.0,), (),
            (), (1.0,), (), (5.0,),
        ),
        kernel=numpy.array((
            (2.0, 0.0,),
            (0.0, 2.0,),
        )),
        stride=numpy.array((2, 2), dtype='int32'),  # h, w
        expected_out_spike_times=(
            (1.0, 3.0,), (),
            (2.0,), (4.0, 6.0),
        )
    )


def run_testcase(
        in_shape,
        in_spike_times,
        kernel,
        stride,
        expected_out_spike_times):
    
    n_input = int(numpy.prod(in_shape))

    run_time = 10.

    sim.setup(timestep=1.)
    sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, (5, 5))
    sim.set_number_of_neurons_per_core(sim.IF_curr_delta, (3, 3))

    src = sim.Population(n_input, sim.SpikeSourceArray,
                         {'spike_times': in_spike_times}, label='input spikes',
                         structure=Grid2D(in_shape[1] / in_shape[0]))

    conn = sim.ConvolutionConnector(kernel, strides=stride)
    out_shape = conn.get_post_shape(in_shape)
    print(out_shape)
    n_output = int(numpy.prod(out_shape))
    print("n_output ", n_output)

    params = {
        'v_thresh': 1.,
        'v_reset': 0.,
        'v': 0.,
        'v_rest': 0.,
        'tau_m': 1.0
    }
    output = sim.Population(n_output, sim.IF_curr_delta, params, label="out",
                            structure=Grid2D(out_shape[1] / out_shape[0]))

    output.record(('spikes',))

    sim.Projection(src, output, conn, sim.Convolution())

    sim.run(run_time)

    neo = output.get_data()

    sim.end()

    actual_out_spike_times = neo.segments[0].spiketrains

    assert(len(actual_out_spike_times) == len(expected_out_spike_times))
    for actual, expected in zip(actual_out_spike_times, expected_out_spike_times):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e


class SingleSpikeKernelResponse(BaseTestCase):

    def check_run(self):
        do_run(plot=False)

    def test_run(self):
        self.runsafe(self.check_run)


if __name__ == '__main__':
    do_run()
