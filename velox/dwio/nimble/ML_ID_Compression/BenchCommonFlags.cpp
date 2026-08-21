/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// Defines the gflags shared across all ML_ID_Compression benchmark drivers.
// Declared in BenchCommon.h; defined here in exactly one TU.

#include <gflags/gflags.h>

DEFINE_string(mlidc_output_csv, "mlidc_results.csv", "CSV output path");
DEFINE_string(
    mlidc_output_manifest,
    "mlidc_manifest.json",
    "JSON manifest sidecar path");
DEFINE_int32(mlidc_rows, 100000, "Number of rows per dataset instance");
DEFINE_int32(mlidc_iters, 5, "Benchmark iterations per (encoder, dataset) pair");
DEFINE_int64(mlidc_seed, 42, "Base random seed for dataset generators");
