/*
 * Copyright (c) 2020 The University of Manchester
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

//! \file
//! \brief Spike processing fast API
#ifndef _SPIKE_PROCESSING_FAST_H_
#define _SPIKE_PROCESSING_FAST_H_

#include <common/neuron-typedefs.h>
#include <common/in_spikes.h>
#include <spin1_api.h>
#include "synapse_row.h"

//! A region of SDRAM used to transfer synapses
struct sdram_config {
    //! The address of the input data to be transferred
    uint32_t *address;
    //! The size of the input data to be transferred
    uint32_t size_in_bytes;
    //! The time of the transfer in us
    uint32_t time_for_transfer_overhead;
};

//! The key and mask being used to send spikes from neurons processed on this
//! core.
struct key_config {
    //! The key
    uint32_t key;
    //! The mask
    uint32_t mask;
    //! The mask to get the spike ID
    uint32_t spike_id_mask;
    //! The colour shift to apply after masking
    uint32_t colour_shift;
    //! Is the node self connected
    uint32_t self_connected;
};

//! Provenance for spike processing
struct spike_processing_fast_provenance {
    //! A count of the times that the synaptic input circular buffers overflowed
    uint32_t n_input_buffer_overflows;
    //! The number of DMAs performed
    uint32_t n_dmas_complete;
    //! The number of spikes received and processed
    uint32_t n_spikes_processed;
    //! The number of rewirings performed.
    uint32_t n_rewires;
    //! The number of packets that were cleared at the end of timesteps
    uint32_t n_packets_dropped_from_lateness;
    //! The maximum size of the input buffer
    uint32_t max_filled_input_buffer_size;
    //! The maximum number of spikes received in a time step
    uint32_t max_spikes_received;
    //! The maximum number of spikes processed in a time step
    uint32_t max_spikes_processed;
    //! The number of times the transfer took longer than expected
    uint32_t n_transfer_timer_overruns;
    //! The number of times a time step was skipped entirely
    uint32_t n_skipped_time_steps;
    //! The maximum additional time taken to transfer
    uint32_t max_transfer_timer_overrun;
    //! The earliest received time of a spike
    uint32_t earliest_receive;
    //! The latest received time of a spike
    uint32_t latest_receive;
    //! The most spikes left at the end of any time step
    uint32_t max_spikes_overflow;
};

//! \brief Set up spike processing
//! \param[in] row_max_n_words The maximum row length in words
//! \param[in] spike_buffer_size The size to make the spike buffer
//! \param[in] discard_late_packets Whether to throw away packets not processed
//!                                 at the end of a time step or keep them for
//!                                 the next time step
//! \param[in] pkts_per_ts_rec_region The ID of the recording region to record
//!                                   packets-per-time-step to
//! \param[in] multicast_priority The priority of multicast processing
//! \param[in] sdram_inputs_param Details of the SDRAM transfer for the ring buffers
//! \param[in] key_config_param Details of the key used by the neuron core
//! \param[in] ring_buffers_param The ring buffers to update with synapse weights
//! \return Whether the setup was successful or not
bool spike_processing_fast_initialise(
        uint32_t row_max_n_words, uint32_t spike_buffer_size,
        bool discard_late_packets, uint32_t pkts_per_ts_rec_region,
        uint32_t multicast_priority, struct sdram_config sdram_inputs_param,
        struct key_config key_config_param, weight_t *ring_buffers_param);

//! \brief The main loop of spike processing to be run once per time step.
//!        Note that this function will not return until the end of the time
//!        step; it will only be interrupted by SDP or MC packets.
//! \param[in] time The time step of the simulation
//! \param[in] n_rewires The number of rewiring attempts to be done
void spike_processing_fast_time_step_loop(uint32_t time, uint32_t n_rewires);

//! \brief Store any provenance data gathered from spike processing
//! \param[in] prov The structure to store the provenance data in
void spike_processing_fast_store_provenance(
        struct spike_processing_fast_provenance *prov);

#endif // _SPIKE_PROCESSING_FAST_H_
