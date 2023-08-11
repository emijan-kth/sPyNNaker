/*
 * Copyright (c) 2021 The University of Manchester
 * based on work Copyright (c) The University of Sussex,
 * Garibaldi Pineda Garcia, James Turner, James Knight and Thomas Nowotny
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
//! \file DTCM-only convolutional processing implementation

#include "local_only_impl.h"
#include "local_only_2d_common.h"
#include <stdlib.h>
#include <debug.h>
#include <stdfix-full-iso.h>
#include "../population_table/population_table.h"
#include "../neuron.h"

// One per connector
typedef struct {
	//! The shape of the kernel
    lc_shape_t kernel;
    //! The shape of the padding
    lc_shape_t padding;
    //! The index of the synapse for positive weights
    uint16_t positive_synapse_type;
    //! The index of the synapse for negative weights
    uint16_t negative_synapse_type;
    uint16_t presynaptic_trace_synapse_type;
	//! The delay stage
	uint16_t delay_stage;
	//! The delay in time steps
    uint16_t delay;
    //! The index of the weights for the kernel
    uint16_t kernel_index;
    //! stride
    lc_coord_t strides;
    //! 1 / stride height
    div_const stride_height_div;
    //! 1 / stride width;
    div_const stride_width_div;
    //! 1 / pooling stride height
    div_const pool_stride_height_div;
    //! 1 / pooling stride width
    div_const pool_stride_width_div;
} connector;

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
} conv_config;

// The main configuration data
static conv_config *config;

static connector *connectors;

static lc_weight_t *weights;

static inline void log_div_const(const char *name, div_const d) {
	log_debug("    %s=(m: %u, sh1: %u, sh2: %u)", name, d.m, d.sh1, d.sh2);
}

//! \brief Load the required data into DTCM.
bool local_only_impl_initialise(void *address){
    log_info("+++++++++++++++++ CONV init ++++++++++++++++++++");
    conv_config* sdram_config = address;
    uint32_t n_bytes = sizeof(conv_config) +
    		(sizeof(source_info) * sdram_config->n_sources);
    config = spin1_malloc(n_bytes);
    if (config == NULL) {
    	log_error("Can't allocate memory for config!");
    	return false;
    }
    spin1_memcpy(config, sdram_config, n_bytes);

    log_info("post_start = %u, %u, post_end = %u, %u, post_shape = %u, %u",
            config->post_start.col, config->post_start.row,
            config->post_end.col, config->post_end.row,
            config->post_shape.width, config->post_shape.height);
    log_info("num sources = %u", config->n_sources);

    if (config->n_sources == 0) {
    	log_error("No sources!");
    	return false;
    }

    // The connectors come after the sources in SDRAM
    connector *sdram_connectors =
    		(connector *) &(sdram_config->sources[config->n_sources]);
    uint32_t n_connector_bytes = sizeof(connector) * config->n_connectors_total;
    connectors = spin1_malloc(n_connector_bytes);
    if (connectors == NULL) {
    	log_error("Can't allocate %u bytes of memory for %u connectors!",
    			n_connector_bytes, config->n_connectors_total);
    	return false;
    }
    spin1_memcpy(connectors, sdram_connectors, n_connector_bytes);

    // The weights come after the connectors in SDRAM
    lc_weight_t *kernel_weights =
    		(lc_weight_t *) &(sdram_connectors[config->n_connectors_total]);
    uint32_t n_weight_bytes = sizeof(lc_weight_t) * config->n_weights_total;
    weights = spin1_malloc(n_weight_bytes);
    if (weights == NULL) {
    	log_error("Can't allocate %u bytes of memory for %u weights!",
    			n_weight_bytes, config->n_weights_total);
    	return false;
    }
    spin1_memcpy(weights, kernel_weights, n_weight_bytes);

    // Print what we have
    for (uint32_t i = 0; i < config->n_sources; i++) {
    	source_info *s_info = &(config->sources[i]);
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
        log_div_const("source_width_div", s_info->source_width_div);
        log_div_const("source_width_last_div", s_info->source_width_last_div);
        log_div_const("cores_per_width_div", s_info->cores_per_width_div);
    }

    for (uint32_t i = 0; i < config->n_connectors_total; i++) {
    	connector *conn = &(connectors[i]);
    	log_debug("Connector %u: kernel size=%u, %u", i, conn->kernel.width,
    			conn->kernel.height);
    	log_debug("    delay=%u, delay_stage=%u", conn->delay, conn->delay_stage);
    }

    return true;
}

//! \brief Calculate the remainder from a division
static inline int16_t calc_remainder(int16_t dividend, int16_t divisor, int16_t quotient) {
    int16_t remainder = dividend - quotient * divisor;
    log_debug("remainder: %d = %d * %d + %d",
            dividend, quotient, divisor, remainder);
    return remainder;
}

//! \brief Do a mapping from pre to post 2D spaces, we use the standard
//! padding, kernel, strides from Convolutional Neural Networks
//! because of the way we're looping through the kernel, we divide the kernel
//! shape by 2.
static inline lc_coord_t map_pre_to_post(connector *connector, lc_coord_t pre, lc_coord_t *start_i) {
    pre.row = div_by_const(pre.row, connector->pool_stride_height_div);
    pre.col = div_by_const(pre.col, connector->pool_stride_width_div);
    pre.row += connector->padding.height;
    pre.col += connector->padding.width;
    lc_coord_t post;
    post.row = div_by_const(pre.row, connector->stride_height_div);
    post.col = div_by_const(pre.col, connector->stride_width_div);
    start_i->row = calc_remainder(pre.row, connector->strides.row, post.row);
    start_i->col = calc_remainder(pre.col, connector->strides.col, post.col);
    return post;
}


//! \brief Given a pre-synaptic coordinate we obtain which post-synaptic
//!        coordinates will be affected (i.e. which of them are 'reached' by
//!        the kernel).
static inline void do_convolution_operation(
        uint32_t time, lc_coord_t pre_coord, connector *connector,
        uint16_t *ring_buffers) {
    lc_coord_t start_i;
    log_debug("kernel height: %d, kernel width: %d, padding height: %d, padding width: %d, strides row: %d, strides col: %d", connector->kernel.height, connector->kernel.width, connector->padding.height, connector->padding.width, connector->strides.row, connector->strides.col);
    lc_coord_t post_coord = map_pre_to_post(connector, pre_coord, &start_i);
    log_debug("pre row %d, col %d AS post row %d, col %d",
            pre_coord.row, pre_coord.col, post_coord.row, post_coord.col);
    lc_weight_t *connector_weights = &weights[connector->kernel_index];

    int32_t kw = connector->kernel.width;
    for (int32_t i_row = start_i.row, tmp_row = post_coord.row; i_row < connector->kernel.height; i_row += connector->strides.row, --tmp_row) {
        int32_t kr = connector->kernel.height - 1 - i_row;
        log_debug("i_row = %u, kr = %u, tmp_row = %u", i_row, kr, tmp_row);

        if ((tmp_row < config->post_start.row) || (tmp_row > config->post_end.row)) {
            log_debug("tmp_row outside");
            continue;
        }

        for (int32_t i_col = start_i.col, tmp_col = post_coord.col; i_col < connector->kernel.width; i_col += connector->strides.col, --tmp_col) {
            int32_t kc = connector->kernel.width - 1 - i_col;

            log_debug("i_col = %u, kc = %u, tmp_col = %u", i_col, kc, tmp_col);
            if ((tmp_col < config->post_start.col) || (tmp_col > config->post_end.col)) {
                log_debug("tmp_col outside");
                continue;
            }

            // This the neuron id relative to the neurons on this core
            uint32_t post_index =
                ((tmp_row - config->post_start.row) * config->post_shape.width)
                    + (tmp_col - config->post_start.col);

            if (connector->presynaptic_trace_synapse_type != 0xffff) {
                uint32_t rb_index = synapse_row_get_ring_buffer_index(time + connector->delay,
                    connector->presynaptic_trace_synapse_type, post_index,
                    synapse_type_index_bits, synapse_index_bits,
                    synapse_delay_mask);
                log_debug("Updating ring_buffers[%u] for post neuron %u = %u, %u, with presynaptic trace",
                        rb_index, post_index, tmp_col, tmp_row);
                // Add one to current ring buffer value, avoiding saturation

                // TODO: Do not use hard coded weight representing 1.0,
                // instead use actual weight scaling
                uint32_t accumulation = ring_buffers[rb_index] + 1;
                uint32_t sat_test = accumulation & 0x10000;
                if (sat_test) {
                    accumulation = sat_test - 1;
                }
                ring_buffers[rb_index] = accumulation;
            }

            uint32_t k = (kr * kw) + kc;
            log_debug("weight index = %u", k);
            lc_weight_t weight = connector_weights[k];
            if (weight == 0) {
                log_debug("zero weight");
                continue;
            }
            uint32_t rb_index = 0;
            if (weight > 0) {
                rb_index = synapse_row_get_ring_buffer_index(time + connector->delay,
                    connector->positive_synapse_type, post_index,
                    synapse_type_index_bits, synapse_index_bits,
                    synapse_delay_mask);
            } else {
                rb_index = synapse_row_get_ring_buffer_index(time + connector->delay,
                    connector->negative_synapse_type, post_index,
                    synapse_type_index_bits, synapse_index_bits,
                    synapse_delay_mask);
                weight = -weight;
            }
            log_debug("Updating ring_buffers[%u] for post neuron %u = %u, %u, with weight %u",
                    rb_index, post_index, tmp_col, tmp_row, weight);

            // Add weight to current ring buffer value, avoiding saturation
            uint32_t accumulation = ring_buffers[rb_index] + weight;
            uint32_t sat_test = accumulation & 0x10000;
            if (sat_test) {
                accumulation = sat_test - 1;
            }
            ring_buffers[rb_index] = accumulation;
        }
    }
}

static inline uint32_t get_core_row(uint32_t core_id, source_info *s_info) {
	return div_by_const(core_id, s_info->cores_per_width_div);
}

static inline uint32_t get_core_col(uint32_t core_id, uint32_t core_row,
		source_info *s_info) {
	return core_id - (core_row * s_info->cores_per_source_width);
}

static inline bool is_last_core_on_row(uint32_t core_col, source_info *s_info) {
	return core_col == (uint32_t) (s_info->cores_per_source_width - 1);
}

static inline bool is_last_core_in_col(uint32_t core_row, source_info *s_info) {
	return core_row == (uint32_t) (s_info->cores_per_source_height - 1);
}

static inline bool key_to_index_lookup(uint32_t spike, source_info **rs_info) {
    for (uint32_t i = 0; i < config->n_sources; i++) {
        source_info *s_info = &(config->sources[i]);
        // We have a match on key
        if ((spike & s_info->key_info.mask) == s_info->key_info.key) {
        	*rs_info = s_info;
        	return true;
        }
    }
    return false;
}

//! \brief Process incoming spikes. In this implementation we need to:
//! 1. Check if it's in the population table
//! 2. Convert the relative (per core) Id to a global (per population) one
//! 3. Obtain the post-ids and weights which will be reached by the spike/kernel
//!    combination.
//! 4. Add the weights to the appropriate current buffers
void local_only_impl_process_spike(
        uint32_t time, uint32_t spike, uint16_t* ring_buffers) {

    // Lookup the spike, and if found, get the appropriate parts
    source_info *s_info;
    if (!key_to_index_lookup(spike, &s_info)) {
    	log_debug("Spike %x didn't match any connectors!", spike);
        return;
    }

    uint32_t core_id = get_core_id(spike, s_info->key_info);
    uint32_t core_row = get_core_row(core_id, s_info);
    uint32_t core_col = get_core_col(core_id, core_row, s_info);
    bool last_core_on_row = is_last_core_on_row(core_col, s_info);
    bool last_core_in_col = is_last_core_in_col(core_row, s_info);
    uint32_t source_height = 0;
    uint32_t source_width = 0;
    div_const source_width_d;
    if (last_core_on_row) {
    	source_width = s_info->source_width_last_core;
    	source_width_d = s_info->source_width_last_div;
    } else {
    	source_width = s_info->source_width_per_core;
    	source_width_d = s_info->source_width_div;
    }
    if (last_core_in_col) {
    	source_height = s_info->source_height_last_core;
    } else {
    	source_height = s_info->source_height_per_core;
    }
    uint32_t local_id = get_local_id(spike, s_info->key_info);
    uint32_t neurons_per_core = source_width * source_height;

    log_debug("Spike %x, on core %u (%u, %u), is last (%u, %u), local %u",
    		spike, core_id, core_col, core_row, last_core_on_row, last_core_in_col,
			local_id);

    // compute the population-based coordinates
    uint32_t end = s_info->key_info.start + s_info->key_info.count;
    for (uint32_t i = s_info->key_info.start; i < end; i++) {
		connector *connector = &(connectors[i]);

    	// Ignore the neuron if the delay does not match
		uint32_t first_neuron = neurons_per_core * connector->delay_stage;
		uint32_t last_neuron = first_neuron + neurons_per_core;
		log_debug("Connector %u, delay stage = %u, first = %u, last = %u, delay = %u",
				i, connector->delay_stage, first_neuron, last_neuron, connector->delay);
    	if (local_id < first_neuron	|| local_id >= last_neuron) {
    		continue;
    	}

    	uint32_t local_neuron_id = local_id - first_neuron;
    	uint32_t local_row = div_by_const(local_neuron_id, source_width_d);
    	uint32_t local_col = local_neuron_id - (local_row * source_width);

    	lc_coord_t pre_coord = {
    	    // The x-coordinate is the remainder of the "division"
    	    .col = (core_col * s_info->source_width_per_core) + local_col,
    	    // The y-coordinate is the integer part of the "division".
    	    .row = (core_row * s_info->source_height_per_core) + local_row
    	};

    	log_debug("Local coord = %u, %u, Pre coord = %u, %u",
    			local_col, local_row, pre_coord.col, pre_coord.row);

		// Compute the convolution
		do_convolution_operation(time, pre_coord, connector, ring_buffers);
    }
}
