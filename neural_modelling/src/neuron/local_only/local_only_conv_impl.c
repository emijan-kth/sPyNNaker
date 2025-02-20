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
#include <stdlib.h>
#include <debug.h>
#include "../population_table/population_table.h"
#include "../neuron.h"

typedef int16_t lc_weight_t;

// Dimensions are needed to be signed due to mapping from pre- to post-synaptic.
typedef int16_t lc_dim_t;

// Reduce the number of parameters with the following structs
typedef struct {
    lc_dim_t row;
    lc_dim_t col;
} lc_coord_t;

typedef struct {
    lc_dim_t height;
    lc_dim_t width;
} lc_shape_t;

typedef struct {
    uint32_t key;
    uint32_t mask;
    uint32_t n_colour_bits;
    uint32_t col_mask;
    uint32_t col_shift;
    uint32_t row_mask;
    uint32_t row_shift;
} source_key_info;

// A reciprocal of a 16-bit signed integer will have 1 sign bit, 1 integer bit
// and 14 fractional bits to allow 1 to be properly represented
#define RECIP_FRACT_BITS 14

// One per connector
typedef struct {
    source_key_info key_info;
    lc_coord_t pre_start;
    lc_shape_t kernel;
    lc_shape_t padding;
    lc_coord_t recip_strides;
    lc_coord_t strides;
    lc_coord_t recip_pool_strides;
    uint16_t positive_synapse_type;
    uint16_t negative_synapse_type;
    uint16_t presynaptic_trace_synapse_type;
    uint16_t unused;
    union {
        uint32_t delay;
        struct {
            uint16_t multisynaptic_delay_max;
            uint16_t multisynaptic_delay_step;
        };
    };
    uint32_t num_multisynaptic_connections;
    uint32_t kernel_index;
} connector;

typedef struct {
    lc_coord_t post_start;
    lc_coord_t post_end;
    lc_shape_t post_shape;
    uint32_t n_weights_total;
    uint32_t n_connectors;
    connector connectors[];
    // In SDRAM, below here is the following:
    // lc_weight_t[] weights;
} conv_config;

// The main configuration data
static conv_config *config;

static lc_weight_t *weights;

//! \brief Load the required data into DTCM.
bool local_only_impl_initialise(void *address){
    log_info("+++++++++++++++++ CONV init ++++++++++++++++++++");
    conv_config* sdram_config = address;
    uint32_t n_bytes = sizeof(conv_config) +
    		(sizeof(connector) * sdram_config->n_connectors);
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
    log_info("num connectors = %u", config->n_connectors);

    if (config->n_connectors == 0) {
    	log_error("No connectors!");
    	return false;
    }

    // The weights come after the config in SDRAM
    lc_weight_t *kernel_weights =
    		(lc_weight_t *) &(sdram_config->connectors[config->n_connectors]);
    uint32_t n_weight_bytes = sizeof(lc_weight_t) * config->n_weights_total;
    weights = spin1_malloc(n_weight_bytes);
    if (weights == NULL) {
    	log_error("Can't allocate memory for weights!");
    	return false;
    }
    spin1_memcpy(weights, kernel_weights, n_weight_bytes);

    // Print what we have
    for (uint32_t i = 0; i < config->n_connectors; i++) {
        log_info("Connector %u: key=0x%08x, mask=0x%08x,"
                "col_mask=0x%08x, col_shift=%u, row_mask=0x%08x, row_shift=%u",
                i, config->connectors[i].key_info.key,
				config->connectors[i].key_info.mask,
				config->connectors[i].key_info.col_mask,
				config->connectors[i].key_info.col_shift,
				config->connectors[i].key_info.row_mask,
				config->connectors[i].key_info.row_shift);
        log_info("              pre_start=%u, %u, kernel_shape=%u %u",
        		config->connectors[i].pre_start.col,
				config->connectors[i].pre_start.row,
				config->connectors[i].kernel.width,
				config->connectors[i].kernel.height);
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

//! \brief Calculate remainder Multiply an integer by a 16-bit reciprocal and return the floored
//!        integer result
static inline int16_t recip_multiply(int16_t integer, int16_t recip) {
    int32_t i = integer;
    int32_t r = recip;
    return (int16_t) ((i * r) >> RECIP_FRACT_BITS);
}

//! \brief Do a mapping from pre to post 2D spaces
static inline lc_coord_t map_pre_to_post(connector *connector, lc_coord_t pre, lc_coord_t *start_i) {
    pre.col = recip_multiply(pre.col, connector->recip_pool_strides.col);
    pre.row = recip_multiply(pre.row, connector->recip_pool_strides.row);
    pre.col += connector->padding.width;
    pre.row += connector->padding.height;
    lc_coord_t post;
    post.col = recip_multiply(pre.col, connector->recip_strides.col);
    post.row = recip_multiply(pre.row, connector->recip_strides.row);
    start_i->col = calc_remainder(pre.col, connector->strides.col, post.col);
    start_i->row = calc_remainder(pre.row, connector->strides.row, post.row);
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

            uint32_t delay;
            if (connector->num_multisynaptic_connections == 1)
            {
                delay = connector->delay;
            }
            else
            {
                kc = connector->kernel.width - 1 - connector->num_multisynaptic_connections * i_col;
                if (kc < 0)
                {
                    log_debug("Multisynaptic connection: i_col = %u, kc = %u, tmp_col = %u", i_col, kc, tmp_col);
                    break;
                }
                delay = connector->multisynaptic_delay_max;
                log_debug("Multisynaptic connection: start_i.col = %u, delay = %u", start_i.col, delay);
            }

            log_debug("i_col = %u, kc = %u, tmp_col = %u", i_col, kc, tmp_col);
            if ((tmp_col < config->post_start.col) || (tmp_col > config->post_end.col)) {
                log_debug("tmp_col outside");
                continue;
            }

            for (int32_t multisynapse_index = 0;
                multisynapse_index < connector->num_multisynaptic_connections;
                ++multisynapse_index, delay -= connector->multisynaptic_delay_step, --kc)
            {
                log_debug("kc = %u, delay = %u", kc, delay);

                // This the neuron id relative to the neurons on this core
                uint32_t post_index =
                    ((tmp_row - config->post_start.row) * config->post_shape.width)
                        + (tmp_col - config->post_start.col);

                if (connector->presynaptic_trace_synapse_type != 0xffff) {
                    uint32_t rb_index = synapse_row_get_ring_buffer_index(time + delay,
                        connector->presynaptic_trace_synapse_type, post_index,
                        synapse_type_index_bits, synapse_index_bits,
                        synapse_delay_mask);
                    log_debug("Updating ring_buffers[%u] for post neuron %u = %u, %u, with presynaptic trace",
                            rb_index, post_index, tmp_col, tmp_row);
                    // Add one to current ring buffer value, avoiding saturation
                    uint32_t accumulation = ring_buffers[rb_index] + 32768;
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
                    rb_index = synapse_row_get_ring_buffer_index(time + delay,
                        connector->positive_synapse_type, post_index,
                        synapse_type_index_bits, synapse_index_bits,
                        synapse_delay_mask);
                } else {
                    rb_index = synapse_row_get_ring_buffer_index(time + delay,
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
}

static inline bool key_to_index_lookup(uint32_t spike, uint32_t *start_index,
		uint32_t *end_index) {
    for (uint32_t i = 0; i < config->n_connectors; i++) {
        connector *c = &(config->connectors[i]);
        if ((spike & c->key_info.mask) == c->key_info.key) {
        	*start_index = i;
            uint32_t e = i + 1;
        	while (e < config->n_connectors) {
        		connector *c_e = &(config->connectors[e]);
        		if ((spike & c_e->key_info.mask) != c_e->key_info.mask) {
        			break;
        		}
        		e = e + 1;
        	}
        	*end_index = e;
        	return true;
        }
    }
    return false;
}

static inline void get_row_col(uint32_t spike, uint32_t index,
		uint32_t *core_local_col, uint32_t *core_local_row) {
	connector *c = &(config->connectors[index]);
	uint32_t local_spike = spike >> c->key_info.n_colour_bits;
	*core_local_col = (local_spike & c->key_info.col_mask) >> c->key_info.col_shift;
	*core_local_row = (local_spike & c->key_info.row_mask) >> c->key_info.row_shift;
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
    uint32_t start;
    uint32_t end;
    if (!key_to_index_lookup(spike, &start, &end)) {
    	log_warning("Spike %u didn't match any connectors!", spike);
        return;
    }
    log_debug("Received spike %u, using connectors between %u and %u", spike, start, end);

    // compute the population-based coordinates
    for (uint32_t i = start; i < end; i++) {
		uint32_t core_local_col;
		uint32_t core_local_row;
		connector *connector = &(config->connectors[i]);
		get_row_col(spike, i, &core_local_col, &core_local_row);
		lc_coord_t pre_coord = {
				core_local_row + connector->pre_start.row,
				core_local_col + connector->pre_start.col
		};
		log_debug("Spike %u = %u, %u (Global: %u, %u)", spike, core_local_col, core_local_row,
				pre_coord.col, pre_coord.row);

		// Compute the convolution
		do_convolution_operation(time, pre_coord, connector, ring_buffers);
    }
}
