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

/*
 * Copyright (c) 2024 by Rivos Inc.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test/generated/algorithms/kernels-metal.h"

// kernel specializations

using namespace breeze::algorithms;

#define _C(X, Y) X##Y
#define C(X, Y) _C(X, Y)

#define NAME(F, T, BT, IPT) C(, F##_##T##_##BT##x##IPT)

#define add_reduce_op ReduceOpAdd
#define min_reduce_op ReduceOpMin
#define max_reduce_op ReduceOpMax

#define GEN_REDUCE_T(O, T, BT, IPT)                                     \
  kernel void NAME(reduce_##O##_##T, T, BT, IPT)(                       \
      const device T *in [[buffer(0)]], device T *out [[buffer(1)]],    \
      const device int *num_items [[buffer(2)]],                        \
      uint thread_idx [[thread_index_in_threadgroup]],                  \
      uint block_idx [[threadgroup_position_in_grid]]) {                \
    MetalPlatform<BT, WARP_THREADS> p{thread_idx, block_idx};           \
    threadgroup typename DeviceReduce<decltype(p), T>::Scratch scratch; \
    reduce<O##_reduce_op, BT, IPT>(p, in, out, &scratch, *num_items);   \
  }

#define GEN_REDUCE(O)         \
  GEN_REDUCE_T(O, int, 32, 2) \
  GEN_REDUCE_T(O, uint, 32, 2)

GEN_REDUCE(add)
GEN_REDUCE(min)
GEN_REDUCE(max)

#define add_scan_op ScanOpAdd

#define GEN_SCAN_T(O, T, BT, IPT, LD)                                          \
  kernel void NAME(scan_##O##_##T, uint_##T, BT, IPT##x##LD)(                  \
      const device T *in [[buffer(0)]], device T *out [[buffer(1)]],           \
      device int *next_block_idx [[buffer(2)]],                                \
      device uint *blocks [[buffer(3)]],                                       \
      const device int *num_items [[buffer(4)]],                               \
      uint thread_idx [[thread_index_in_threadgroup]],                         \
      uint block_idx [[threadgroup_position_in_grid]]) {                       \
    MetalPlatform<BT, WARP_THREADS> p{thread_idx, block_idx};                  \
    threadgroup typename DeviceScan<decltype(p), T, IPT, LD>::Scratch scratch; \
    scan<O##_scan_op, BT, IPT, LD>(p, in, out, next_block_idx, blocks,         \
                                   &scratch, *num_items);                      \
  }

#define GEN_SCAN(O)              \
  GEN_SCAN_T(O, int, 32, 2, 32)  \
  GEN_SCAN_T(O, int, 64, 2, 64)  \
  GEN_SCAN_T(O, uint, 32, 2, 32) \
  GEN_SCAN_T(O, uint, 64, 2, 64)

GEN_SCAN(add)

#define GEN_RADIX_SORT_HISTOGRAM_T(T, BT, IPT, TS, RB)                   \
  kernel void NAME(radix_sort_histogram, T, BT, IPT##x##TS##x##RB)(      \
      const device T *in [[buffer(0)]], device uint *out [[buffer(1)]],  \
      const device int *num_items [[buffer(2)]],                         \
      uint thread_idx [[thread_index_in_threadgroup]],                   \
      uint block_idx [[threadgroup_position_in_grid]]) {                 \
    threadgroup DeviceRadixSortHistogram<RB, T>::Scratch scratch;        \
    radix_sort_histogram<BT, IPT, TS, RB>(                               \
        MetalPlatform<BT, WARP_THREADS>{thread_idx, block_idx}, in, out, \
        &scratch, *num_items);                                           \
  }

#define GEN_RADIX_SORT_HISTOGRAM(T)          \
  GEN_RADIX_SORT_HISTOGRAM_T(T, 64, 2, 1, 6) \
  GEN_RADIX_SORT_HISTOGRAM_T(T, 64, 2, 4, 6)

GEN_RADIX_SORT_HISTOGRAM(int)
GEN_RADIX_SORT_HISTOGRAM(uint)

#define GEN_RADIX_SORT(T, BT, IPT, RB)                                        \
  kernel void NAME(radix_sort, T, BT, IPT##x##RB)(                            \
      const device T *in [[buffer(0)]],                                       \
      const device uint *in_offsets [[buffer(1)]],                            \
      const device int *start_bit [[buffer(2)]],                              \
      const device int *num_pass_bits [[buffer(3)]],                          \
      device T *out [[buffer(4)]], device int *next_block_idx [[buffer(5)]],  \
      device uint *blocks [[buffer(6)]],                                      \
      const device int *num_items [[buffer(7)]],                              \
      uint thread_idx [[thread_index_in_threadgroup]],                        \
      uint block_idx [[threadgroup_position_in_grid]]) {                      \
    MetalPlatform<BT, WARP_THREADS> p{thread_idx, block_idx};                 \
    threadgroup DeviceRadixSort<decltype(p), IPT, RB, T>::Scratch scratch;    \
    radix_sort<BT, IPT, RB>(p, in, in_offsets, start_bit, num_pass_bits, out, \
                            next_block_idx, blocks, &scratch, *num_items);    \
  }

GEN_RADIX_SORT(int, 64, 2, 6)
GEN_RADIX_SORT(uint, 64, 2, 6)
