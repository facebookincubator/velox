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

// FFI declarations for the lance-c library (liblance_c.a).
// These match the C API defined in lance-c's include/lance.h.

extern "C" {

// --- Opaque handles ---

typedef struct LanceDataset LanceDataset;
typedef struct LanceScanner LanceScanner;
typedef struct LanceBatch LanceBatch;

// --- Error handling ---

typedef enum {
  LANCE_OK = 0,
  LANCE_ERR_INVALID_ARGUMENT = 1,
  LANCE_ERR_IO = 2,
  LANCE_ERR_NOT_FOUND = 3,
  LANCE_ERR_DATASET_ALREADY_EXISTS = 4,
  LANCE_ERR_INDEX = 5,
  LANCE_ERR_INTERNAL = 6,
  LANCE_ERR_NOT_SUPPORTED = 7,
  LANCE_ERR_COMMIT_CONFLICT = 8,
} LanceErrorCode;

LanceErrorCode lance_last_error_code(void);
const char* lance_last_error_message(void);
void lance_free_string(const char* s);

// --- Dataset lifecycle ---

LanceDataset* lance_dataset_open(
    const char* uri,
    const char* const* storage_opts,
    uint64_t version);

void lance_dataset_close(LanceDataset* dataset);

// --- Dataset metadata ---

uint64_t lance_dataset_version(const LanceDataset* dataset);
uint64_t lance_dataset_count_rows(const LanceDataset* dataset);
int32_t lance_dataset_schema(const LanceDataset* dataset, ArrowSchema* out);

// --- Fragment enumeration ---

uint64_t lance_dataset_fragment_count(const LanceDataset* dataset);
int32_t lance_dataset_fragment_ids(
    const LanceDataset* dataset,
    uint64_t* out_ids);

// --- Scanner builder ---

LanceScanner* lance_scanner_new(
    const LanceDataset* dataset,
    const char* const* columns,
    const char* filter);

int32_t lance_scanner_set_limit(LanceScanner* scanner, int64_t limit);
int32_t lance_scanner_set_offset(LanceScanner* scanner, int64_t offset);
int32_t lance_scanner_set_batch_size(
    LanceScanner* scanner,
    int64_t batch_size);
int32_t lance_scanner_set_fragment_ids(
    LanceScanner* scanner,
    const uint64_t* ids,
    size_t len);

void lance_scanner_close(LanceScanner* scanner);

// --- Sync scan: batch iteration ---

/// Returns: 0 = batch available, 1 = end of stream, -1 = error
int32_t lance_scanner_next(LanceScanner* scanner, LanceBatch** out);

// --- Batch (Arrow C Data Interface) ---

int32_t lance_batch_to_arrow(
    const LanceBatch* batch,
    ArrowArray* out_array,
    ArrowSchema* out_schema);

void lance_batch_free(LanceBatch* batch);

} // extern "C"
