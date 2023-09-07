/*
 * Copyright (c) 2017 The University of Manchester
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "in_spikes_with_payload.h"

#include <common/send_mc.h>

#include <data_specification.h>
#include <debug.h>
#include <simulation.h>
#include <stdbool.h>

// ----------------------------------------------------------------------

//! The keys to be used by the neurons (one per neuron)
uint32_t *neuron_keys;

//! A checker that says if this model should be transmitting. If set to false
//! by the data region, then this model should not have a key.
bool use_key;

//! Latest time in a timestep that any neuron has sent a spike
uint32_t latest_send_time = 0xFFFFFFFF;

//! Earliest time in a timestep that any neuron has sent a spike
uint32_t earliest_send_time = 0;

//! The colour of the time step to handle delayed spikes
uint32_t colour = 0;

//! The number of neurons on the core
static uint32_t n_neurons;

//! The closest power of 2 >= n_neurons
static uint32_t n_neurons_peak;

//! The number of synapse types
static uint32_t n_synapse_types;

//! The mask of the colour
static uint32_t colour_mask;

//! Amount to left shift the ring buffer by to make it an input
static uint32_t *ring_buffer_to_input_left_shifts;

//! parameters that reside in the neuron_parameter_data_region
struct neuron_core_parameters {
    uint32_t has_key;
    uint32_t n_neurons_to_simulate;
    uint32_t n_neurons_peak;
    uint32_t n_colour_bits;
    uint32_t n_synapse_types;
    uint32_t ring_buffer_shifts[];
    // Following this struct in memory (as it can't be expressed in C) is:
    // uint32_t neuron_keys[n_neurons_to_simulate];
};

//! The provenance information written on application shutdown.
struct WTA_provenance {
    //! A count of the times that the synaptic input circular buffers overflowed
    uint32_t n_input_buffer_overflows;
};

//! Mask for selecting the neuron ID from a spike
#define NEURON_ID_MASK     0x7FF

// Globals
//! The simulation time
static uint32_t time;
//! Current simulation stop/pause time
static uint32_t simulation_ticks;
//! True if the simulation is running continuously
static uint32_t infinite_run;

//! DSG regions in use
enum regions {
    SYSTEM_REGION,
    CORE_PARAMS_REGION,
    PROVENANCE_DATA_REGION,
    PROFILER_REGION,
    RECORDING_REGION,
    NEURON_PARAMS_REGION,
    CURRENT_SOURCE_PARAMS_REGION,
    NEURON_RECORDING_REGION,
    SDRAM_PARAMS_REGION,
	INITIAL_VALUES_REGION
};

//! values for the priority for each callback
enum WTA_callback_priorities {
    MC = -1,   //!< Multicast message reception is FIQ
    SDP = 0,   //!< SDP handling is highest normal priority
    DMA = 1,   //!< DMA complete handling is medium priority
    TIMER = 2, //!< Timer interrupt processing is lowest priority
};

// ----------------------------------------------------------------------

// //! \brief Send a SpiNNaker multicast-with-payload message to the motor hardware
// //! \param[in] direction: Which direction to move in
// //! \param[in] the_speed: What speed to move at
// static inline void send_to_motor(uint32_t direction, uint32_t the_speed) {
//     uint32_t direction_key = direction | key;
//     while (!spin1_send_mc_packet(direction_key, the_speed, WITH_PAYLOAD)) {
//         spin1_delay_us(1);
//     }
//     if (delay_time > 0) {
//         spin1_delay_us(delay_time);
//     }
// }

// Callbacks
//! \brief Regular 1ms callback. Takes spikes from circular buffer and process.
//! \param unused0: unused
//! \param unused1: unused
static void timer_callback(UNUSED uint unused0, UNUSED uint unused1) {
    time++;

    log_debug("Timer tick %d", time);

    if (simulation_is_finished()) {
        simulation_handle_pause_resume(NULL);
        log_info("Simulation complete.\n");
        simulation_ready_to_read();
    }

    // Process the incoming spikes

    REAL max_membrane_voltage = 0.0;
    uint32_t max_neuron = UINT32_MAX;

    spike_t spike;
    uint32_t nid;

    union
    {
        REAL membrane_voltage;
        uint32_t payload;
    } s;

    while (in_spikes_get_next_spike(&spike)) {
        // TODO: Calculate this properly
        nid = (spike_key(spike) & NEURON_ID_MASK);
        s.payload = spike_payload(spike);

        log_debug("Received spike from neuron %x, membrane voltage: %12.6k", nid, s.membrane_voltage);

        if (s.membrane_voltage > max_membrane_voltage)
        {
            max_membrane_voltage = s.membrane_voltage;
            max_neuron = nid;
        }
    }

    // Check if we received any spikes
    if (max_neuron != UINT32_MAX)
    {
        // Process the spike with highest membrane voltage
        log_debug("Spike with highest membrane voltage was received from neuron %x, membrane voltage: %12.6k", max_neuron, max_membrane_voltage);

        if (use_key)
        {
            // TODO: Calculate this properly
            uint32_t spike_key = neuron_keys[0] + max_neuron;

            log_debug("Sending spike with key %x", spike_key);

            send_spike_mc(spike_key);
        }
    }
}

//! \brief Reads the configuration
//! \param[in] config_region: Where to read the configuration from
static bool read_parameters(struct neuron_core_parameters *config_region) {
    log_info("Reading parameters from 0x%.8x", config_region);

    // Check if there is a key to use
    use_key = config_region->has_key;

    // Read the neuron details
    n_neurons = config_region->n_neurons_to_simulate;
    n_neurons_peak = config_region->n_neurons_peak;
    n_synapse_types = config_region->n_synapse_types;

    // Get colour details
    colour_mask = (1 << config_region->n_colour_bits) - 1;


    // The key list comes after the ring buffer shifts
    uint32_t *neuron_keys_sdram =
            (uint32_t *) &config_region->ring_buffer_shifts[n_synapse_types];
    uint32_t neuron_keys_size = n_neurons * sizeof(uint32_t);
    neuron_keys = spin1_malloc(neuron_keys_size);
    if (neuron_keys == NULL) {
        log_error("Not enough memory to allocate neuron keys");
        return false;
    }
    spin1_memcpy(neuron_keys, neuron_keys_sdram, neuron_keys_size);


    // Allocate the space for the schedule
    // counters = spin1_malloc(N_COUNTERS * sizeof(int));
    // last_speed = spin1_malloc(N_COUNTERS * sizeof(int));

    // for (uint32_t i = 0; i < N_COUNTERS; i++) {
    //     counters[i] = 0;
    //     last_speed[i] = 0;
    // }

    for (uint32_t i = 0; i < n_neurons; ++i)
    {
        log_info("Key %d = %x", i, neuron_keys[i]);
    }

    return true;
}

//! \brief Add incoming spike message (in FIQ) to circular buffer
//! \param[in] key: The received spike
//! \param payload: ignored
static void incoming_spike_callback_payload(uint key, uint payload) {
    log_debug("Received spike %x at time %d with payload %12.6k", key, time, payload);

    union _spike_t spike = { .key = key, .payload = payload };

    in_spikes_add_spike(spike.pair);

    in_spikes_print_buffer();
}

//! \brief Callback to store provenance data (format: neuron_provenance).
//! \param[out] provenance_region: Where to write the provenance data
static void c_main_store_provenance_data(address_t provenance_region) {
    log_debug("writing other provenance data");
    struct WTA_provenance *prov = (void *) provenance_region;

    // store the data into the provenance data region
    prov->n_input_buffer_overflows = in_spikes_get_n_buffer_overflows();

    log_debug("finished other provenance data");
}

//! \brief Read all application configuration
//! \param[out] timer_period: How long to program ticks to be
//! \return True if initialisation succeeded
static bool initialize(uint32_t *timer_period) {
    log_info("initialise: started");

    // Get the address this core's DTCM data starts at from SRAM
    data_specification_metadata_t *ds_regions =
            data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(ds_regions)) {
        return false;
    }

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM_REGION, ds_regions),
            APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
            &infinite_run, &time, SDP, DMA)) {
        return false;
    }

    simulation_set_provenance_function(
        c_main_store_provenance_data,
        data_specification_get_region(PROVENANCE_DATA_REGION, ds_regions));

    // Get the parameters
    if (!read_parameters(data_specification_get_region(CORE_PARAMS_REGION, ds_regions)))
    {
        return false;
    }

    log_info("initialise: completed successfully");

    return true;
}

//! Entry point
void c_main(void) {
    // Initialise
    uint32_t timer_period = 0;
    if (!initialize(&timer_period)) {
        log_error("Error in initialisation - exiting!");
        rt_error(RTE_SWERR);
    }

    // Initialise the incoming spike buffer
    if (!in_spikes_initialize_spike_buffer(8192)) {
        return;
    }

    // Set timer_callback
    spin1_set_timer_tick(timer_period);

    // Register callbacks
    spin1_callback_on(
        MCPL_PACKET_RECEIVED, incoming_spike_callback_payload, MC);
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;
    simulation_run();
}
