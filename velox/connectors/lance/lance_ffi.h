/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "velox/vector/arrow/Abi.h"

// FFI declarations for the lance-duckdb Rust library (liblance_duckdb_ffi.a).
// These are a subset of the functions from the lance-duckdb project needed for
// reading Lance datasets.

extern "C" {

/// Opens a Lance dataset at the given path. Returns an opaque dataset handle,
/// or nullptr on error. Check lance_last_error_message() on failure.
void* lance_open_dataset(const char* path);

/// Closes and frees the dataset handle.
void lance_close_dataset(void* dataset);

/// Returns an opaque schema handle for the dataset's scan schema.
/// Returns nullptr on error.
void* lance_get_schema_for_scan(void* dataset);

/// Converts an opaque schema handle to an ArrowSchema.
/// Returns 0 on success, non-zero on error.
int32_t lance_schema_to_arrow(void* schema, ArrowSchema* out);

/// Frees an opaque schema handle.
void lance_free_schema(void* schema);

/// Creates a scan stream over the dataset.
/// @param dataset Opaque dataset handle.
/// @param columns Array of column name strings to project (nullptr = all).
/// @param columns_len Number of column names.
/// @param filter_ir Filter IR bytes (nullptr = no filter).
/// @param filter_ir_len Length of filter_ir.
/// @param limit Max rows to return (-1 = unlimited).
/// @param offset Row offset to start from (-1 = 0).
/// @return Opaque stream handle, or nullptr on error.
void* lance_create_dataset_stream_ir(
    void* dataset,
    const char** columns,
    size_t columns_len,
    const uint8_t* filter_ir,
    size_t filter_ir_len,
    int64_t limit,
    int64_t offset);

/// Gets the next batch from the stream.
/// @param stream Opaque stream handle.
/// @param out_batch Pointer to receive the opaque batch handle.
/// @return 0 if a batch was returned, 1 if the stream is exhausted,
///         negative on error.
int32_t lance_stream_next(void* stream, void** out_batch);

/// Closes and frees a stream handle.
void lance_close_stream(void* stream);

/// Converts an opaque batch handle to Arrow C Data Interface structs.
/// @return 0 on success, non-zero on error.
int32_t
lance_batch_to_arrow(void* batch, ArrowArray* out_array, ArrowSchema* out_schema);

/// Frees an opaque batch handle.
void lance_free_batch(void* batch);

/// Returns the error code from the last failed FFI call.
int32_t lance_last_error_code();

/// Returns the error message from the last failed FFI call.
/// The returned string is valid until the next FFI call on the same thread.
const char* lance_last_error_message();

/// Frees a string allocated by the FFI library.
void lance_free_string(char* s);

} // extern "C"
