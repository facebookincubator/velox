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

// Copyright (c) 2024 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "breeze/algorithms/reduce.h"
#include "breeze/algorithms/scan.h"
#include "breeze/algorithms/sort.h"
#include "breeze/platforms/openmp.h"
#include "breeze/platforms/platform.h"
#include "breeze/utils/types.h"

namespace kernels {

template <typename Op, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename U, typename SharedMemType,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void Reduce(PlatformT p, SharedMemType* scratch, const T* in, U* out,
            int num_items) {
  breeze::algorithms::DeviceReduce<PlatformT, U>::template Reduce<
      Op, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out),
      breeze::utils::make_slice<breeze::utils::SHARED>(scratch), num_items);
}

template <typename Op, int BLOCK_THREADS, int ITEMS_PER_THREAD,
          int LOOKBACK_DISTANCE, typename T, typename U, typename V,
          typename SharedMemType,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void Scan(PlatformT p, SharedMemType* scratch, const T* in, U* out,
          int* next_block_idx, V* blocks, int num_items) {
  breeze::algorithms::DeviceScan<PlatformT, U, ITEMS_PER_THREAD,
                                 LOOKBACK_DISTANCE>::
      template Scan<Op>(
          p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
          breeze::utils::make_slice<breeze::utils::GLOBAL>(out),
          breeze::utils::make_slice<breeze::utils::GLOBAL>(next_block_idx),
          breeze::utils::make_slice<breeze::utils::GLOBAL>(blocks),
          breeze::utils::make_slice<breeze::utils::SHARED>(scratch), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int TILE_SIZE,
          int RADIX_BITS, typename T, typename SharedMemType,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void RadixSortHistogram(PlatformT p, SharedMemType* scratch, const T* in,
                        unsigned* out, int num_items) {
  breeze::algorithms::DeviceRadixSortHistogram<RADIX_BITS, T>::template Build<
      ITEMS_PER_THREAD, TILE_SIZE>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out),
      breeze::utils::make_slice<breeze::utils::SHARED>(scratch), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int RADIX_BITS, typename T,
          typename SharedMemType,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void RadixSort(PlatformT p, SharedMemType* scratch, const T* in,
               const unsigned* in_offsets, const int* start_bit,
               const int* num_pass_bits, T* out, int* next_block_idx,
               unsigned* blocks, int num_items) {
  breeze::algorithms::DeviceRadixSort<PlatformT, ITEMS_PER_THREAD, RADIX_BITS,
                                      T>::
      template Sort<unsigned>(
          p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
          breeze::utils::make_slice<breeze::utils::GLOBAL>(in_offsets),
          *start_bit, *num_pass_bits,
          breeze::utils::make_slice<breeze::utils::GLOBAL>(out),
          breeze::utils::make_slice<breeze::utils::GLOBAL>(next_block_idx),
          breeze::utils::make_slice<breeze::utils::GLOBAL>(blocks),
          breeze::utils::make_slice<breeze::utils::SHARED>(scratch), num_items);
}

}  // namespace kernels
