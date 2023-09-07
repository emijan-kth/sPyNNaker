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

#include "../neuron/local_only/local_only_2d_common.h"

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


typedef struct {
	//! Information about the key
	key_info key_info;
	//! The source population height per core
	uint32_t source_height_per_core: 16;
	//! The source population width per core
	uint32_t source_width_per_core: 16;
	//! The source population height on the last core in a column
	uint32_t source_height_last_core: 16;
	//! The source population width on the last core on a row
	uint32_t source_width_last_core: 16;
	//! The number cores in a height of the source
	uint32_t cores_per_source_height: 16;
    //! Number of cores in a width of the source
    uint32_t cores_per_source_width: 16;
	//! Used to calculate division by the source width per core efficiently
    div_const source_width_div;
    //! Division by last core width
    div_const source_width_last_div;
    //! Division by cores per source width
    div_const cores_per_width_div;
} source_info;


typedef struct {
    lc_coord_t post_start;
    lc_coord_t post_end;
    lc_shape_t post_shape;
    uint32_t n_sources;
    uint32_t n_connectors_total;
    uint32_t n_weights_total;
    source_info sources[];
    // In SDRAM, after sources[n_sources] is the following:
    // connector connectors[n_connectors_total];
    // lc_weight_t[n_weights_total] weights;
} ConvolutionConfig;


// The main configuration data
static ConvolutionConfig *convolution_config;


int16_t num_neurons_in;

REAL *max_membrane_voltages;
uint32_t *max_source_indices;



//! The provenance information provided by neurons
struct neuron_provenance {
    //! The current time.
    uint32_t current_timer_tick;
    //! The number of times a TDMA slot was missed
    uint32_t n_tdma_misses;
    //! Earliest send time within any time step
    uint32_t earliest_send;
    //! Latest send time within any time step
    uint32_t latest_send;
};

//: Provenance data for local-only processing
struct local_only_provenance {
	//! The maximum number of spikes received in a time step
    uint32_t max_spikes_received_per_timestep;
    //! The number of spikes dropped due to running out of time in a time step
    uint32_t n_spikes_dropped;
    //! The number of spikes dropped due to the queue having no space
    uint32_t n_spikes_lost_from_input;
    //! The maximum size of the spike input queue at any time
    uint32_t max_input_buffer_size;
};

//! The combined provenance from synapses and neurons
struct combined_provenance {
    struct neuron_provenance neuron_provenance;
    struct local_only_provenance local_only_provenance;
    //! Maximum backgrounds queued
    uint32_t max_backgrounds_queued;
    //! Background queue overloads
    uint32_t n_background_queue_overloads;
};

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
    PROVENANCE_DATA_REGION,
    PROFILER_REGION,
    RECORDING_REGION,
	CORE_PARAMS_REGION,
    NEURON_PARAMS_REGION,
    CURRENT_SOURCE_PARAMS_REGION,
    NEURON_RECORDING_REGION,
    LOCAL_ONLY_REGION,
    LOCAL_ONLY_PARAMS_REGION,
	NEURON_BUILDER_REGION,
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

static inline bool key_to_index_lookup(uint32_t key, uint32_t *source_index) {
    for (uint32_t i = 0; i < convolution_config->n_sources; i++) {
        source_info *s_info = &(convolution_config->sources[i]);
        // We have a match on key
        if ((key & s_info->key_info.mask) == s_info->key_info.key) {
            *source_index = i;
        	return true;
        }
    }
    return false;
}

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

    for (int16_t i = 0; i < num_neurons_in; ++i)
    {
        max_membrane_voltages[i] = 0.0;
        max_source_indices[i] = UINT32_MAX;
    }

    spike_t spike;

    union
    {
        REAL membrane_voltage;
        uint32_t payload;
    } s;

    while (in_spikes_get_next_spike(&spike)) {

        key_t key = spike_key(spike);

        log_debug("Received spike with key %x", key);

        // Lookup the spike, and if found, get the appropriate parts
        uint32_t source_index;
        if (!key_to_index_lookup(key, &source_index)) {
            log_debug("Spike with key %x didn't match any connectors!", key);
            return;
        }

        source_info *s_info = &(convolution_config->sources[source_index]);

        uint32_t channel = key & s_info->key_info.mask;

        //uint32_t neuron_id = key & ~s_info->key_info.mask;

        uint32_t neuron_id = get_local_id(key, s_info->key_info);

        s.payload = spike_payload(spike);   

        log_debug(
            "Spike with key %x has channel: %x, source: %d, and neuron_id: %x. "
            "Payload is: %12.6k",
            key,
            channel,
            source_index,
            neuron_id,
            s.membrane_voltage);

        if (s.membrane_voltage > max_membrane_voltages[neuron_id])
        {
            max_membrane_voltages[neuron_id] = s.membrane_voltage;
            max_source_indices[neuron_id] = source_index;
        }
    }

    // For each input neuron, check if we received any spikes
    for (int16_t neuron_id = 0; neuron_id < num_neurons_in; ++neuron_id)
    {
        uint32_t source_index = max_source_indices[neuron_id];
        
        if (source_index != UINT32_MAX)
        {
            // Process the spike with highest membrane voltage
            log_debug(
                "For neuron_id: %d, "
                "spike with highest membrane voltage was received from source %d, "
                "membrane voltage: %12.6k",
                neuron_id,
                source_index,
                max_membrane_voltages[neuron_id]);

            if (use_key)
            {
                uint32_t neuron_id_out = source_index * num_neurons_in + neuron_id;
                uint32_t key = neuron_keys[neuron_id_out];

                log_debug(
                    "Sending spike with outgoing neuron_id %x, "
                    " key: %x",
                    neuron_id_out,
                    key);

                send_spike_mc(key);
            }
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

    for (uint32_t i = 0; i < n_neurons; ++i)
    {
        log_info("Key %d = %x", i, neuron_keys[i]);
    }

    return true;
}

bool local_only_initialise(void *address) {

    log_info("+++++++++++++++++ CONV init ++++++++++++++++++++");
    ConvolutionConfig* sdram_config = address;
    uint32_t n_bytes = sizeof(ConvolutionConfig) +
    		(sizeof(source_info) * sdram_config->n_sources);
    convolution_config = spin1_malloc(n_bytes);
    if (convolution_config == NULL) {
    	log_error("Can't allocate memory for config!");
    	return false;
    }
    spin1_memcpy(convolution_config, sdram_config, n_bytes);

    log_info("post_start = %u, %u, post_end = %u, %u, post_shape = %u, %u",
            convolution_config->post_start.col, convolution_config->post_start.row,
            convolution_config->post_end.col, convolution_config->post_end.row,
            convolution_config->post_shape.width, convolution_config->post_shape.height);
    log_info("num sources = %u", convolution_config->n_sources);

    if (convolution_config->n_sources == 0) {
    	log_error("No sources!");
    	return false;
    }

    // Allocate memory for variables used during spike processing

    source_info *s_info_0 = &(convolution_config->sources[0]);
    num_neurons_in = s_info_0->source_height_per_core * s_info_0->source_width_per_core;

    max_membrane_voltages = spin1_malloc(num_neurons_in * sizeof(max_membrane_voltages[0]));
    max_source_indices = spin1_malloc(num_neurons_in * sizeof(max_source_indices[0]));

    if (max_membrane_voltages == NULL || max_source_indices == NULL) {
        log_error("Not enough memory");
        return false;
    }

    // Print what we have
    for (uint32_t i = 0; i < convolution_config->n_sources; i++) {
    	source_info *s_info = &(convolution_config->sources[i]);
        log_debug("Source %u: key=0x%08x, mask=0x%08x, start=%u, count=%u",
                i, s_info->key_info.key, s_info->key_info.mask,
				s_info->key_info.start, s_info->key_info.count);
        log_debug("    core_mask=0x%08x, mask_shift=0x%08x",
        		s_info->key_info.core_mask, s_info->key_info.mask_shift);
        log_debug("    height_per_core=%u, width_per_core=%u",
        		s_info->source_height_per_core, s_info->source_width_per_core);
        log_debug("    height_last_core=%u, width_last_core=%u",
        		s_info->source_height_last_core, s_info->source_width_last_core);
        log_debug("    cores_per_height=%u, cores_per_width=%u",
        		s_info->cores_per_source_height, s_info->cores_per_source_width);
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

static inline void store_neuron_provenance(struct neuron_provenance *prov) {
    prov->current_timer_tick = time;
    prov->n_tdma_misses = 0;
    prov->earliest_send = earliest_send_time;
    prov->latest_send = latest_send_time;
}

static inline void local_only_store_provenance(struct local_only_provenance *prov) {
    prov->max_spikes_received_per_timestep = 0;
    prov->n_spikes_dropped = 0;
    prov->n_spikes_lost_from_input = in_spikes_get_n_buffer_overflows();
    prov->max_input_buffer_size = 0;
}

//! \brief Callback to store provenance data (format: neuron_provenance).
//! \param[out] provenance_region: Where to write the provenance data
static void c_main_store_provenance_data(address_t provenance_region) {
    log_debug("writing other provenance data");

    struct combined_provenance *prov = (void *) provenance_region;
    prov->n_background_queue_overloads = 0;
    prov->max_backgrounds_queued = 0;
    store_neuron_provenance(&prov->neuron_provenance);
    local_only_store_provenance(&prov->local_only_provenance);

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

    if (!local_only_initialise(
            data_specification_get_region(LOCAL_ONLY_PARAMS_REGION, ds_regions))) {
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
