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

#ifndef _LOCAL_ONLY_CONV_IMPL_H_
#define _LOCAL_ONLY_CONV_IMPL_H_

#include "local_only_impl.h"
#include "local_only_2d_common.h"

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
} local_only_conv_source_info;

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
} local_only_conv_connector;

typedef struct {
    lc_coord_t post_start;
    lc_coord_t post_end;
    lc_shape_t post_shape;
    uint32_t n_sources;
    uint32_t n_connectors_total;
    uint32_t n_weights_total;
    local_only_conv_source_info sources[];
    // In SDRAM, after sources[n_sources] is the following:
    // connector connectors[n_connectors_total];
    // lc_weight_t[n_weights_total] weights;
} local_only_conv_config_t;

#endif
