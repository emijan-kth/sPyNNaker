# Multiple fixes to convolutions in SpiNNaker/sPyNNaker*

## Overview

I have identified possible defects in the convolution implementation in sPyNNaker (ConvolutionConnector in the Python API, including underlying C implementation).

The possible defects include:

- Problems using convolution kernels of arbitrary size, in particular non-square kernels and kernels where the height and/or width are non-odd numbers.
- Incorrect behavior when convolutional strides are used, in particular non-square strides.
- Documentation of parameters to ConvolutionConnector does not agree with actual implementation, in regard to (row, column) ordering.


## Draft corrections

I have developed draft corrections to all identified defects. The corrections can be found in my sPyNNaker fork on Github: 

[Suggested corrections to sPyNNaker](https://github.com/SpiNNakerManchester/sPyNNaker/compare/master...emijan-kth:sPyNNaker:master)


## Some details on the defects

### Reading of convolutional kernel weights from outside the weight array

It seems like the implementation sometimes reads weights from outside the weight array, i.e. on line

https://github.com/SpiNNakerManchester/sPyNNaker/blob/064c620841529694ef419dae627c21842148e98b/neural_modelling/src/neuron/local_only/local_only_conv_impl.c#L195

I have tried to test this by using a minimal case of a 2x1 convolution kernel: (1.0, 2.0), applied on an input shape of 3x1.
At t=0 there is a single spike at the center neuron (x=1, y=0). See below for a complete Python program.

The post shape is correctly calculated as 2x1.

Also, the weight applied on the spike when calculating the input to the first post neuron (x=0, y=0) is 2.0, which seems to be correct.

However, the weight applied on the spike when calculating the input to the second post neuron (x=1, y=0) is 1.43, which seems strange to me.

To understand what is happening, I have added some debug outputs to `local_only_conv_impl.c`, printing the values of some of the variables.
Then I can see that kc=2, which will make k=2. As far as I can understand, that will make the program reading a weight from outside the weight array,
as the length of the weight array should be 2 in this case. Here, the value at `connector->weights[2]` happens to be 5844:

```
[DEBUG] (local_only_conv_impl.c: 271): Received spike 16 = 1, 0 (Global: 1, 0)
[DEBUG] (local_only_conv_impl.c: 176): pre row 0, col 1 AS post row 0, col 0
[DEBUG] (local_only_conv_impl.c: 180): xxx1: r = 0, kr = 0
[DEBUG] (local_only_conv_impl.c: 183): xxx1: tmp_row = 0
[DEBUG] (local_only_conv_impl.c: 189): xxx2: c = 4294967295, kc = 0
[DEBUG] (local_only_conv_impl.c: 191): xxx2: tmp_col = 4294967295
[DEBUG] (local_only_conv_impl.c: 189): xxx2: c = 0, kc = 1
[DEBUG] (local_only_conv_impl.c: 191): xxx2: tmp_col = 0
[DEBUG] (local_only_conv_impl.c: 219): Updating ring_buffers[4] for post neuron 0 = 0, 0, with weight 8192
[DEBUG] (local_only_conv_impl.c: 189): xxx2: c = 1, kc = 2
[DEBUG] (local_only_conv_impl.c: 191): xxx2: tmp_col = 1
[DEBUG] (local_only_conv_impl.c: 219): Updating ring_buffers[5] for post neuron 1 = 1, 0, with weight 5844
[DEBUG] (c_main_local_only.c: 145): Inputs
[DEBUG] (neuron_impl_standard.h: 423): Neuron   0: input    2.000000 (= 
   2.000000 -    0.000000925
[DEBUG] (neuron_impl_standard.h: 423): Neuron   1: input    1.426758 (= 
   1.426758 -    0.000000925
```

The reason for this seems to be that the inner loop iterates from -half_kw up to half_kw (inclusive).
Thus it looks like the inner loop assumes that the kernel width is 3, when it actually only is 2.

Below is a complete test program, which produces the following output:

`y=0:	x=0:{}{0.000 0.000 0.363 0.297 }	x=1:{}{0.000 0.000 0.259 0.212 }`	

**Test program:**

```
import numpy as np

import pyNN.spiNNaker as pynn
from pyNN.space import Grid2D

def main():
    kernel = np.array(
        (
            (1.0, 2.0,),
        ),
    )

    strides = (1, 1)

    input_spikes = np.array((
        ((), (0.0,), (),),
    ), dtype=object)

    run(kernel, strides, input_spikes)


def get_neuron():
    neuron_type_class = pynn.IF_curr_delta

    return neuron_type_class(
        tau_m=5.0,
        cm=5.0,
        v_rest=0.0,
        v_reset=0.0,
        v_thresh=0.5,
        tau_refrac=1.0,
        i_offset=0.0,
        v=0.0,
        isyn_exc=0.0,
        isyn_inh=0.0
    )


def run(kernel, strides, input_spikes):
    pynn.setup(timestep=1)

    neuron_type = get_neuron()

    connector = pynn.ConvolutionConnector(kernel.T, strides=strides)

    flattened_input = input_spikes.reshape(-1)

    spike_source = pynn.SpikeSourceArray(spike_times=flattened_input)

    input_height, input_width = input_spikes.shape
    num_neurons = input_height * input_width

    input_neurons = pynn.Population(
        num_neurons,
        spike_source,
        label='Input neurons',
        structure=Grid2D(input_width / input_height))

    input_shape = input_width, input_height
    output_width, output_height = connector.get_post_shape(input_shape)
    num_output_neurons = output_width * output_height

    output_neurons = pynn.Population(
        num_output_neurons,
        neuron_type,
        label='Output neurons',
        structure=Grid2D(output_width / output_height))

    pynn.Projection(input_neurons, output_neurons, connector, pynn.Convolution())

    output_neurons.record(('spikes', 'v'))

    duration = 4
    pynn.run(duration)

    output_data = output_neurons.get_data()

    pynn.end()

    visualize_output(output_data, output_width, output_height)


def visualize_output(output_data, output_width, output_height):
    segments = output_data.segments[0]
    spiketrains = segments.spiketrains
    membrane_voltage = segments.filter(name="v")[0]

    for y in range(output_height):
        print(f"y={y}:\t", end='')
        for x in range(output_width):
            neuron_index = y * output_width + x
            spiketrain = spiketrains[neuron_index]
            print(f"x={x}:{{", end='')
            for t in spiketrain:
                print(f"{int(t) }", end='')
            print("}{", end='')
            for v in membrane_voltage[:, neuron_index]:
                print(f"{float(v):.3f}", end=' ')
            print("}\t", end='')
        print()


if __name__ == "__main__":
    main()
```


### Strange results using strides

Changing the beginning of the test program above to the following:

```
    kernel = np.array(
        (
            (3.0, 2.0, 1.0),
        ),
    )

    strides = (2, 1)

    input_spikes = np.array((
        ((), (), (0.0,), (), ()),
    ), dtype=object)
```

i.e.

- a 3x1 kernel (odd width kernel to avoid the possible bug above)
- a stride of 2x1
- an input shape of 5x1 with a single input spike at t=0.0 at the center neuron (x=2, y=0)

First, to get the output shape right I had to do the following change to convolution_connector.py to even be able to run the program:

https://github.com/emijan-kth/sPyNNaker/commit/1cd8e621cf59d9bcbee37c9c802d3aadd0023936

After this change, the post shape is calculated as 2x1.

**Now, I would expect this to apply weights 3.0 and 1.0 when calculating the inputs to the first and to the second post neurons, respectively.**

**However, this actually applies weights 2.0 and 1.0:**

```
[DEBUG] (c_main_local_only.c: 157): Timer tick 0 

[INFO] (neuron_model_lif_impl.h: 193): V membrane    =     0.0000 mv
[INFO] (neuron_model_lif_impl.h: 194): Refract timer = 0 timesteps
[INFO] (neuron_model_lif_impl.h: 193): V membrane    =     0.0000 mv
[INFO] (neuron_model_lif_impl.h: 194): Refract timer = 0 timesteps
[DEBUG] (local_only_conv_impl.c: 271): Received spike 32 = 2, 0 (Global: 2, 0)
[DEBUG] (local_only_conv_impl.c: 176): pre row 0, col 2 AS post row 0, col 0
[DEBUG] (local_only_conv_impl.c: 180): xxx1: r = 0, kr = 0
[DEBUG] (local_only_conv_impl.c: 183): xxx1: tmp_row = 0
[DEBUG] (local_only_conv_impl.c: 189): xxx2: c = 4294967295, kc = 0
[DEBUG] (local_only_conv_impl.c: 191): xxx2: tmp_col = 4294967295
[DEBUG] (local_only_conv_impl.c: 189): xxx2: c = 0, kc = 1
[DEBUG] (local_only_conv_impl.c: 191): xxx2: tmp_col = 0
[DEBUG] (local_only_conv_impl.c: 219): Updating ring_buffers[4] for post neuron 0 = 0, 0, with weight 4096
[DEBUG] (local_only_conv_impl.c: 189): xxx2: c = 1, kc = 2
[DEBUG] (local_only_conv_impl.c: 191): xxx2: tmp_col = 1
[DEBUG] (local_only_conv_impl.c: 219): Updating ring_buffers[5] for post neuron 1 = 1, 0, with weight 2048
[DEBUG] (c_main_local_only.c: 145): Inputs
[DEBUG] (neuron_impl_standard.h: 423): Neuron   0: input    2.000000 (= 
   2.000000 -    0.000000925
[DEBUG] (neuron_impl_standard.h: 423): Neuron   1: input    1.000000 (= 
   1.000000 -    0.000000925
```

and thus produces output:

`y=0:	x=0:{}{0.000 0.000 0.363 0.297 0.243 }	x=1:{}{0.000 0.000 0.181 0.148 0.122 }`
