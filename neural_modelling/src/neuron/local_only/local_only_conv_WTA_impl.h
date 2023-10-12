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

#ifndef _LOCAL_ONLY_H_
#define _LOCAL_ONLY_H_

#include <common/neuron-typedefs.h>
#include <stdlib.h>
#include <debug.h>

#include "local_only_2d_common.h"

extern uint32_t synapse_delay_mask;

extern uint32_t synapse_type_index_bits;

extern uint32_t synapse_index_bits;

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
    //! True if source is WTA reset
    bool is_WTA_reset;
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

extern conv_config *local_only_conv_config;

//! \brief Load the data required
//! \return Whether the loading was successful or not
bool local_only_impl_initialise(void *address);

//! \brief Process a spike received
void local_only_impl_process_spike(uint32_t time, uint32_t spike,
        uint16_t* ring_buffers);

static inline bool is_key_WTA_reset(uint32_t spike) {
    for (uint32_t i = 0; i < local_only_conv_config->n_sources; i++) {
        source_info *s_info = &(local_only_conv_config->sources[i]);
        if ((spike & s_info->key_info.mask) == s_info->key_info.key) {
            // We have a match on key
            log_debug("Matched source: %d", i);
            return s_info->is_WTA_reset;
        }
    }
    return false;
}

#endif
