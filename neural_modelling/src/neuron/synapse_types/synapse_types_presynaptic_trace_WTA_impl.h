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

/*!
 * \file
 * \brief implementation of synapse_types.h for synapse with presynaptic trace
 * and WTA reset input.
 *
 * If we have combined excitatory/inhibitory synapses it will be
 * because both excitatory and inhibitory synaptic time-constants
 * (and thus propagators) are identical.
 */

#ifndef _SYNAPSE_TYPES_PRESYNAPTIC_TRACE_IMPL_H_
#define _SYNAPSE_TYPES_PRESYNAPTIC_TRACE_IMPL_H_

//---------------------------------------
// Macros
//---------------------------------------
//! \brief Number of bits to encode the synapse type
//! \details <tt>ceil(log2(#SYNAPSE_TYPE_COUNT))</tt>
#define SYNAPSE_TYPE_BITS 2
//! \brief Number of synapse types
//! \details <tt>#NUM_EXCITATORY_RECEPTORS + #NUM_INHIBITORY_RECEPTORS</tt>
#define SYNAPSE_TYPE_COUNT 4

//! Number of excitatory receptors
#define NUM_EXCITATORY_RECEPTORS 1
//! Number of inhibitory receptors
#define NUM_INHIBITORY_RECEPTORS 2

#include <debug.h>
#include <common/neuron-typedefs.h>
#include "synapse_types.h"
#include "exp_synapse_utils.h"

//---------------------------------------
// Synapse parameters
//---------------------------------------
struct synapse_types_params_t {
    input_t exc;
    input_t inh;
    exp_params_t trace;
    REAL alpha;
    REAL time_step_ms;
};

struct synapse_types_t {
    input_t exc; //!< Excitatory synaptic input
    input_t inh; //!< Inhibitory synaptic input
    input_t trace; //!< Presynaptic trace input
    decay_t trace_decay;
    decay_t trace_input_factor;
    bool WTA_reset; //!< WTA reset input
};

//! The supported synapse type indices
typedef enum {
    EXCITATORY, //!< Excitatory synaptic input
    INHIBITORY, //!< Inhibitory synaptic input
    TRACE, //!< Presynaptic trace input
    WTA_RESET, //!< WTA reset input
} synapse_presynaptic_trace_input_buffer_regions;

//---------------------------------------
// Synapse shaping inline implementation
//---------------------------------------

static inline void synapse_types_initialise(synapse_types_t *state,
		synapse_types_params_t *params, UNUSED uint32_t n_steps_per_timestep) {
	state->exc = params->exc;
	state->inh = params->inh;
    state->WTA_reset = false;

	REAL ts = kdivui(params->time_step_ms, n_steps_per_timestep);
	REAL ts_over_tau = kdivk(ts, params->trace.tau);

    state->trace_decay = 1.0ulr - ts_over_tau;
    state->trace_input_factor = ts_over_tau * params->alpha;
	state->trace = params->trace.init_input;

	log_debug("state->trace_decay = %11.4k", (REAL) state->trace_decay);
	log_debug("state->trace_input_factor = %11.4k", (REAL) state->trace_input_factor);
}

static void synapse_types_save_state(synapse_types_t *state, synapse_types_params_t *params) {
	params->exc = state->exc;
	params->inh = state->inh;
	params->trace.init_input = state->trace;
}

//! \brief decays the stuff thats sitting in the input buffers as these have not
//!     yet been processed and applied to the neuron.
//!
//! This is to compensate for the valve behaviour of a synapse in biology
//! (spike goes in, synapse opens, then closes slowly)
//! plus the leaky aspect of a neuron.
//!
//! \param[in,out] parameters: the pointer to the parameters to use
static inline void synapse_types_shape_input(
        synapse_types_t *parameters) {
	parameters->exc = ZERO;
	parameters->inh = ZERO;
    parameters->WTA_reset = false;
    log_debug("Shape input, before: parameters->trace = %11.4k", parameters->trace);
    parameters->trace = decay_s1615(parameters->trace, parameters->trace_decay);
    log_debug("Shape input, after: parameters->trace = %11.4k", parameters->trace);
}

//! \brief adds the inputs for a give timer period to a given neuron that is
//!     being simulated by this model
//! \param[in] synapse_type_index the type of input that this input is to be
//!     considered (aka excitatory or inhibitory etc)
//! \param[in,out] parameters: the pointer to the parameters to use
//! \param[in] input the inputs for that given synapse_type.
static inline void synapse_types_add_neuron_input(
        index_t synapse_type_index, synapse_types_t *parameters,
        input_t input) {
    log_debug("synapse_type_index = %u, input = %11.4k", synapse_type_index, input);
    switch (synapse_type_index) {
    case EXCITATORY:
    	parameters->exc += input;
    	break;
    case INHIBITORY:
    	parameters->inh += input;
    	break;
    case TRACE:
        log_debug(
            "Add neuron input, before: trace.synaptic_input_value = %11.4k, input = %11.4k",
            parameters->trace,
            input);
        parameters->trace = parameters->trace + decay_s1615(input, parameters->trace_input_factor);
        log_debug("Add neuron input, after: trace.synaptic_input_value = %11.4k", parameters->trace);
    	break;
    case WTA_RESET:
        log_debug("Received WTA reset input");
        parameters->WTA_reset = true;
        break;
    }
}

//! \brief extracts the excitatory input buffers from the buffers available
//!     for a given parameter set
//! \param[in,out] excitatory_response: Buffer to put response in
//! \param[in] parameters: the pointer to the parameters to use
//! \return the excitatory input buffers for a given neuron ID.
static inline input_t *synapse_types_get_excitatory_input(
        input_t *excitatory_response, synapse_types_t *parameters) {
    excitatory_response[0] = parameters->exc;
    return &excitatory_response[0];
}

//! \brief extracts the inhibitory input buffers from the buffers available
//!     for a given parameter set
//! \param[in,out] inhibitory_response: Buffer to put response in
//! \param[in] parameters: the pointer to the parameters to use
//! \return the inhibitory input buffers for a given neuron ID.
static inline input_t *synapse_types_get_inhibitory_input(
        input_t *inhibitory_response, synapse_types_t *parameters) {
    inhibitory_response[0] = parameters->inh;
    inhibitory_response[1] = parameters->trace;
    return &inhibitory_response[0];
}

static inline bool synapse_types_get_reset_input(
        synapse_types_t *parameters) {
    return parameters->WTA_reset;
}

//! \brief returns a human readable character for the type of synapse.
//!     examples would be X = excitatory types, I = inhibitory types etc etc.
//! \param[in] synapse_type_index: the synapse type index
//!     (there is a specific index interpretation in each synapse type)
//! \return a human readable character representing the synapse type.
static inline const char *synapse_types_get_type_char(
        index_t synapse_type_index) {
    switch (synapse_type_index) {
    case EXCITATORY:
        return "X";
    case INHIBITORY:
        return "I";
    case TRACE:
        return "T";
    case WTA_RESET:
        return "W";
    default:
        log_debug("did not recognise synapse type %i", synapse_type_index);
        return "?";
    }
}

//! \brief prints the input for a neuron ID given the available inputs
//!     currently only executed when the models are in debug mode, as the prints
//!     are controlled from the synapses.c print_inputs() method.
//! \param[in] parameters: the pointer to the parameters to use
static inline void synapse_types_print_input(
        synapse_types_t *parameters) {
    io_printf(IO_BUF, "%12.6k - %12.6k - %12.6k - %d",
            parameters->exc, parameters->inh, parameters->trace, parameters->WTA_reset);
}

//! \brief printer call
//! \param[in] parameters: the pointer to the parameters to print
static inline void synapse_types_print_parameters(
        UNUSED synapse_types_t *parameters) {
    log_info("trace_decay  = %11.4k", parameters->trace_decay);
    log_info("trace_input_factor   = %11.4k", parameters->trace_input_factor);
    log_info("gsyn_trace_initial_value = %11.4k",
            parameters->trace);
}

#endif  // _SYNAPSE_TYPES_PRESYNAPTIC_TRACE_IMPL_H_
