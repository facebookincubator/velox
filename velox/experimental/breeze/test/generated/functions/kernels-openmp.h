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

#include "breeze/functions/load.h"
#include "breeze/functions/reduce.h"
#include "breeze/functions/scan.h"
#include "breeze/functions/sort.h"
#include "breeze/functions/store.h"
#include "breeze/platforms/openmp.h"
#include "breeze/platforms/platform.h"

namespace kernels {

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockLoad(PlatformT p, const T* in, T* out, int num_items) {
  T items[ITEMS_PER_THREAD];
  breeze::functions::BlockLoad<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice(items), num_items);
  breeze::functions::BlockStore<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice(items),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockLoadIf(PlatformT p, const T* in, const int* in_selection_flags,
                 T* out, int num_items) {
  int selection_flags[ITEMS_PER_THREAD];
  breeze::functions::BlockLoad<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in_selection_flags),
      breeze::utils::make_slice(selection_flags), num_items);
  T items[ITEMS_PER_THREAD];
  breeze::functions::BlockLoadIf<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice(selection_flags),
      breeze::utils::make_slice(items), num_items);
  breeze::functions::BlockStoreIf<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice(items),
      breeze::utils::make_slice(selection_flags),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockLoadFrom(PlatformT p, const T* in, const int* in_offsets, T* out,
                   int num_items) {
  int offsets[ITEMS_PER_THREAD];
  breeze::functions::BlockLoad<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in_offsets),
      breeze::utils::make_slice(offsets), num_items);
  T items[ITEMS_PER_THREAD];
  breeze::functions::BlockLoadFrom<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice(offsets), breeze::utils::make_slice(items),
      num_items);
  breeze::functions::BlockStore<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice(items),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockStore(PlatformT p, const T* in, T* out, int num_items) {
  breeze::functions::BlockStore<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p,
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          in),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockStoreIf(PlatformT p, const T* in, const int* selection_flags, T* out,
                  int num_items) {
  breeze::functions::BlockStoreIf<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p,
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          in),
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          selection_flags),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockStoreAt(PlatformT p, const T* in, const int* offsets, T* out,
                  int num_items) {
  breeze::functions::BlockStoreAt<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p,
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          in),
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          offsets),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockStoreAtIf(PlatformT p, const T* in, const int* offsets,
                    const int* selection_flags, T* out, int num_items) {
  breeze::functions::BlockStoreAtIf<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p,
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          in),
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          offsets),
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          selection_flags),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockFill(PlatformT p, const T* value, T* out, int num_items) {
  breeze::functions::BlockFill<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, *value, breeze::utils::make_slice<breeze::utils::GLOBAL>(out),
      num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockFillAtIf(PlatformT p, const T* value, const int* offsets,
                   const int* selection_flags, T* out, int num_items) {
  breeze::functions::BlockFillAtIf<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, *value,
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          offsets),
      breeze::utils::make_slice<breeze::utils::GLOBAL, breeze::utils::STRIPED>(
          selection_flags),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <typename Op, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename U, typename SharedMemType,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockReduce(PlatformT p, SharedMemType* scratch, const T* in, U* out,
                 int num_items) {
  T items[ITEMS_PER_THREAD];
  breeze::functions::BlockLoad<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice(items), num_items);
  U aggregate = breeze::functions::BlockReduce<PlatformT, U>::template Reduce<
      Op, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice(items),
      breeze::utils::make_slice<breeze::utils::SHARED>(scratch), num_items);
  p.syncthreads();
  if (p.thread_idx() == 0) {
    *out = aggregate;
  }
}

template <typename Op, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename U, typename SharedMemType,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockScan(PlatformT p, SharedMemType* scratch, const T* in, U* out,
               int num_items) {
  T items[ITEMS_PER_THREAD];
  breeze::functions::BlockLoad<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice(items), num_items);
  U sums[ITEMS_PER_THREAD];
  breeze::functions::BlockScan<PlatformT, U, ITEMS_PER_THREAD>::template Scan<
      Op>(p, breeze::utils::make_slice(items), breeze::utils::make_slice(sums),
          breeze::utils::make_slice<breeze::utils::SHARED>(scratch), num_items);
  breeze::functions::BlockStore<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice(sums),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int RADIX_BITS, typename T,
          typename SharedMemType,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockRadixRank(PlatformT p, SharedMemType* scratch, const T* in, int* out,
                    int num_items) {
  T items[ITEMS_PER_THREAD];
  // initialize invalid items to max value
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    items[i] = static_cast<T>((1 << RADIX_BITS) - 1);
  }
  breeze::functions::BlockLoad<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice<breeze::utils::THREAD,
                                breeze::utils::WARP_STRIPED>(items),
      num_items);
  int ranks[ITEMS_PER_THREAD];
  breeze::functions::BlockRadixRank<PlatformT, ITEMS_PER_THREAD, RADIX_BITS>::
      Rank(p,
           breeze::utils::make_slice<breeze::utils::THREAD,
                                     breeze::utils::WARP_STRIPED>(items),
           breeze::utils::make_slice<breeze::utils::THREAD,
                                     breeze::utils::WARP_STRIPED>(ranks),
           breeze::utils::make_slice<breeze::utils::SHARED>(scratch));
  breeze::functions::BlockStore<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p,
      breeze::utils::make_slice<breeze::utils::THREAD,
                                breeze::utils::WARP_STRIPED>(ranks),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int RADIX_BITS, typename T,
          typename SharedMemType,
          typename PlatformT = OpenMPPlatform<BLOCK_THREADS, BLOCK_THREADS>>
void BlockRadixSort(PlatformT p, SharedMemType* scratch, const T* in, T* out,
                    int num_items) {
  T items[ITEMS_PER_THREAD];
  breeze::functions::BlockLoad<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice<breeze::utils::THREAD,
                                breeze::utils::WARP_STRIPED>(items),
      num_items);
  breeze::functions::
      BlockRadixSort<PlatformT, ITEMS_PER_THREAD, RADIX_BITS, T>::Sort(
          p,
          breeze::utils::make_slice<breeze::utils::THREAD,
                                    breeze::utils::WARP_STRIPED>(items),
          breeze::utils::make_slice<breeze::utils::SHARED>(scratch), num_items);
  breeze::functions::BlockStore<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p,
      breeze::utils::make_slice<breeze::utils::THREAD,
                                breeze::utils::WARP_STRIPED>(items),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out), num_items);
}

}  // namespace kernels
