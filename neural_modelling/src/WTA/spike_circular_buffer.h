/*
 * Copyright (c) 2013 The University of Manchester
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
//! \brief Fast circular buffer
#ifndef _CIRCULAR_BUFFER_H_
#define _CIRCULAR_BUFFER_H_

#include <stdint.h>
#include <stdbool.h>

#include "spin-print.h"
#include "utils.h"


//! Implementation of a circular buffer
typedef struct _spike_circular_buffer {
    //! The size of the buffer. One less than a power of two.
    uint32_t buffer_size;
    //! The index of the next position in the buffer to read from.
    uint32_t output;
    //! The index of the next position in the buffer to write to.
    uint32_t input;
    //! \brief The number of times an insertion has failed due to the buffer
    //!     being full.
    uint32_t overflows;
    //! The buffer itself.
    BUFFER_ITEM_TYPE buffer[];
} _spike_circular_buffer;

//! The public interface type is a pointer to the implementation
typedef _spike_circular_buffer *spike_circular_buffer;

//! \brief Get the index of the next position in the buffer from the given
//!     value.
//! \param[in] buffer: The buffer.
//! \param[in] current: The index to get the next one after.
//! \return The next index after the given one.
static inline uint32_t spike_circular_buffer_next(
        spike_circular_buffer buffer, uint32_t current) {
    return (current + 1) & buffer->buffer_size;
}

//! \brief Get whether the buffer is not empty.
//! \param[in] buffer: The buffer.
//! \return Whether the buffer has anything in it.
static inline bool spike_circular_buffer_not_empty(spike_circular_buffer buffer) {
    return buffer->input != buffer->output;
}

//! \brief Get whether the buffer is able to accept more values
//! \param[in] buffer: The buffer.
//! \param[in] next: The next position in the buffer
//! \return Whether the buffer has room to take another value.
static inline bool spike_circular_buffer_not_full(
        spike_circular_buffer buffer, uint32_t next) {
    return next != buffer->output;
}

//! \brief Create a new FIFO circular buffer of at least the given size. For
//!     efficiency, the buffer can be bigger than requested.
//! \param[in] size: The minimum number of elements in the buffer to be created
//! \return A reference to the created buffer.
static spike_circular_buffer spike_circular_buffer_initialize(uint32_t size);

//! \brief Add an item to an existing buffer.
//! \param[in] buffer: The buffer struct to add to
//! \param[in] item: The item to add.
//! \return Whether the item was added (it fails if the buffer is full).
static inline bool spike_circular_buffer_add(spike_circular_buffer buffer, BUFFER_ITEM_TYPE item) {
    uint32_t next = spike_circular_buffer_next(buffer, buffer->input);
    bool success = spike_circular_buffer_not_full(buffer, next);

    if (success) {
	    buffer->buffer[buffer->input] = item;
	    buffer->input = next;
    } else {
	    buffer->overflows++;
    }

    return success;
}

//! \brief Get the next item from an existing buffer.
//! \param[in] buffer: The buffer to get the next item from.
//! \param[out] item: The retrieved item.
//! \return Whether an item was retrieved.
static inline bool spike_circular_buffer_get_next(
        spike_circular_buffer buffer, BUFFER_ITEM_TYPE *item) {
    bool success = spike_circular_buffer_not_empty(buffer);

    if (success) {
	    *item = buffer->buffer[buffer->output];
	    buffer->output = spike_circular_buffer_next(buffer, buffer->output);
    }

    return success;
}

//! \brief Advance the buffer if the next item is equal to the given value.
//! \param[in] buffer: The buffer to advance
//! \param[in] item: The item to check
//! \return Whether the buffer was advanced.
static inline bool spike_circular_buffer_advance_if_next_equals(
        spike_circular_buffer buffer, BUFFER_ITEM_TYPE item) {
    bool success = spike_circular_buffer_not_empty(buffer);
    if (success) {
	    success = (buffer->buffer[buffer->output] == item);
	    if (success) {
	        buffer->output = spike_circular_buffer_next(buffer, buffer->output);
	    }
    }
    return success;
}

//! \brief Get the size of the buffer.
//! \param[in] buffer: The buffer to get the size of
//! \return The number of elements currently in the buffer
static inline uint32_t spike_circular_buffer_size(spike_circular_buffer buffer) {
    return buffer->input >= buffer->output
	    ? buffer->input - buffer->output
	    : (buffer->input + buffer->buffer_size + 1) - buffer->output;
}

//! \brief Get the number of overflows that have occurred when adding to
//!     the buffer.
//! \param[in] buffer: The buffer to check for overflows
//! \return The number of times add was called and returned False
static inline uint32_t spike_circular_buffer_get_n_buffer_overflows(
	    spike_circular_buffer buffer) {
    return buffer->overflows;
}

//! \brief Clear the circular buffer.
//! \param[in] buffer: The buffer to clear
static inline void spike_circular_buffer_clear(spike_circular_buffer buffer) {
    buffer->input = 0;
    buffer->output = 0;
}

//! \brief Print the contents of the buffer.
//! \details Do not use if the sark `IO_BUF` is being used for binary data.
//! \param[in] buffer: The buffer to print
static void spike_circular_buffer_print_buffer(spike_circular_buffer buffer);

//---------------------------------------
// Synaptic rewiring support functions
//---------------------------------------
//! \brief Get the input index.
//! \param[in] buffer: The buffer.
//! \return The index that the next value to be put into the buffer will be
//!     placed at.
static inline uint32_t spike_circular_buffer_input(spike_circular_buffer buffer) {
    return buffer->input;
}

//! \brief Get the output index.
//! \param[in] buffer: The buffer.
//! \return The index that the next value to be removed from the buffer
//!     is/will be at.
static inline uint32_t spike_circular_buffer_output(spike_circular_buffer buffer) {
    return buffer->output;
}

//! \brief Get the buffer size.
//! \param[in] buffer: The buffer.
//! \return The real size of the buffer.
static inline uint32_t spike_circular_buffer_real_size(spike_circular_buffer buffer) {
    return buffer->buffer_size;
}

//! \brief Get the buffer contents at a particular index.
//! \param[in] buffer: The buffer.
//! \param[in] index: The index to use. Note that the index is not limited to
//!     the size of the buffer.
//! \return The contents of the buffer at a particular index.
static inline BUFFER_ITEM_TYPE spike_circular_buffer_value_at_index(
        spike_circular_buffer buffer, uint32_t index) {
    return buffer->buffer[index & buffer->buffer_size];
}

static spike_circular_buffer spike_circular_buffer_initialize(
	uint32_t size)
{
    uint32_t real_size = size;
    if (!is_power_of_2(real_size)) {
	real_size = next_power_of_2(size);
    }

    spike_circular_buffer buffer = sark_alloc(1,
	    sizeof(_spike_circular_buffer) + real_size * sizeof(uint32_t));
    if (buffer == NULL) {
	return NULL;
    }

    buffer->buffer_size = real_size - 1;
    buffer->input = 0;
    buffer->output = 0;
    buffer->overflows = 0;
    return buffer;
}

static void spike_circular_buffer_print_buffer(
	spike_circular_buffer buffer)
{
    uint32_t i = buffer->output;

    io_printf(IO_BUF, "[");
    while (i != buffer->input) {

    #if BUFFER_ITEM_TYPE == spike_t && defined(SPIKES_WITH_PAYLOADS)
	    io_printf(IO_BUF, "(%u:%u)", spike_key(buffer->buffer[i]), spike_payload(buffer->buffer[i]));
    #else
	    io_printf(IO_BUF, "%u", buffer->buffer[i]);
    #endif
	i = (i + 1) & buffer->buffer_size;
	if (i != buffer->input) {
	    io_printf(IO_BUF, ", ");
	}
    }
    io_printf(IO_BUF, "]\n");
}

#endif // _CIRCULAR_BUFFER_H_
