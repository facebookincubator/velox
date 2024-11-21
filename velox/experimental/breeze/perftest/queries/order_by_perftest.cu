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

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "breeze/algorithms/sort.h"
#include "breeze/functions/load.h"
#include "breeze/functions/reduce.h"
#include "breeze/functions/scan.h"
#include "breeze/functions/store.h"
#include "breeze/platforms/cuda.cuh"
#include "breeze/utils/device_vector.h"
#include "breeze/utils/types.h"
#include "perftest/perftest.h"
#include "perftest/queries/caching_device_allocator.cuh"
#include "perftest/queries/device_column.h"

namespace breeze {
namespace test {
namespace kernels {

enum { CUDA_WARP_THREADS = 32 };

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int TILE_SIZE,
          int RADIX_BITS, typename T, typename U>
__global__ __launch_bounds__(BLOCK_THREADS) void BuildRadixSortHistogram(
    T* in_buffers[2], const int* in_selector, U* histogram, int num_items) {
  using namespace algorithms;
  using namespace utils;

  CudaPlatform<BLOCK_THREADS, CUDA_WARP_THREADS> p;
  using RadixSortHistogramT = DeviceRadixSortHistogram<RADIX_BITS, T>;
  __shared__ typename RadixSortHistogramT::Scratch scratch;

  const T* in = in_buffers[*in_selector];
  RadixSortHistogramT::template Build<ITEMS_PER_THREAD, TILE_SIZE>(
      p, make_slice<GLOBAL>(in), make_slice<GLOBAL>(histogram),
      make_slice(&scratch).template reinterpret<SHARED>(), num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int RADIX_BITS, typename U>
__global__
__launch_bounds__(BLOCK_THREADS) void RadixSortHistogramExclusiveScan(
    const U* in, U* out, int* buffer_advancement) {
  using namespace functions;
  using namespace utils;

  enum {
    NUM_BINS = 1 << RADIX_BITS,
  };

  CudaPlatform<BLOCK_THREADS, CUDA_WARP_THREADS> p;
  using BlockReduceT = BlockReduce<decltype(p), U>;
  using BlockScanT = BlockScan<decltype(p), U, ITEMS_PER_THREAD>;
  __shared__ struct {
    typename BlockReduceT::Scratch reduce_sum;
    typename BlockReduceT::Scratch reduce_max;
    typename BlockScanT::Scratch scan;
  } scratch;

  U items[ITEMS_PER_THREAD];

  // load counts
  const U* it = in + p.block_idx() * NUM_BINS;
  BlockLoad<BLOCK_THREADS, ITEMS_PER_THREAD>(p, make_slice<GLOBAL>(it),
                                             make_slice(items), NUM_BINS);

  // reductions to determine if all items are in the same bin
  U sum = BlockReduceT::template Reduce<ReduceOpAdd, ITEMS_PER_THREAD>(
      p, make_slice(items), make_slice<SHARED>(&scratch.reduce_sum), NUM_BINS);
  U max = BlockReduceT::template Reduce<ReduceOpMax, ITEMS_PER_THREAD>(
      p, make_slice(items), make_slice<SHARED>(&scratch.reduce_max), NUM_BINS);

  // advance buffer unless all items are in the same bin (sum == max) and
  // pass can be skipped
  if (p.thread_idx() == 0) {
    buffer_advancement[p.block_idx()] = sum == max ? 0 : 1;
  }

  // inclusive scan
  U offsets[ITEMS_PER_THREAD];
  BlockScanT::template Scan<ScanOpAdd>(
      p, make_slice(items), make_slice(offsets),
      make_slice<SHARED>(&scratch.scan), NUM_BINS);

  // convert inclusive scan to exclusive scan
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    offsets[i] -= items[i];
  }

  // store results
  U* out_it = out + p.block_idx() * NUM_BINS;
  BlockStore<BLOCK_THREADS, ITEMS_PER_THREAD>(
      p, make_slice(offsets), make_slice<GLOBAL>(out_it), NUM_BINS);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_THREADS) void UpdateBufferSelectors(
    const int* buffer_advancements, int* input_selector,
    int* buffer_selectors) {
  // get initial index from input selector
  int current_selector = *input_selector;

  // we have 2 selectors per pass; first selector determines the input and
  // second determines the output
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    buffer_selectors[i * 2] = current_selector;
    current_selector = (current_selector + buffer_advancements[i]) % 2;
    buffer_selectors[i * 2 + 1] = current_selector;
  }

  // update input selector to the final output selector
  *input_selector = current_selector;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int RADIX_BITS, typename T,
          typename U, typename BlockT>
__global__ __launch_bounds__(BLOCK_THREADS) void RadixSort(
    const int* in_buffer_selectors, const U* in_offsets, int start_bit,
    int num_pass_bits, T* buffers[2], int* next_block_idx, BlockT* blocks,
    int num_items) {
  using namespace algorithms;
  using namespace utils;

  CudaPlatform<BLOCK_THREADS, CUDA_WARP_THREADS> p;
  using RadixSortT =
      DeviceRadixSort<decltype(p), ITEMS_PER_THREAD, RADIX_BITS, T>;
  __shared__ typename RadixSortT::Scratch scratch;

  // load buffer selectors
  int current_selector = in_buffer_selectors[0];
  int alternate_selector = in_buffer_selectors[1];

  // sorting pass is only needed if input and output selectors are different
  if (current_selector != alternate_selector) {
    const T* in = buffers[current_selector];
    T* out = buffers[alternate_selector];

    RadixSortT::template Sort<BlockT>(
        p, make_slice<GLOBAL>(in), make_slice<GLOBAL>(in_offsets), start_bit,
        num_pass_bits, make_slice<GLOBAL>(out),
        make_slice<GLOBAL>(next_block_idx), make_slice<GLOBAL>(blocks),
        make_slice(&scratch).template reinterpret<SHARED>(), num_items);
  }
}

}  // namespace kernels

using OrderByConfig = PerfTestArrayConfig<11>;

const OrderByConfig kConfig = {{
    {"num_key_rows", "400000"},
    {"num_key_rows_short", "6400"},
    {"num_key_rows_grande", "6400000"},
    {"num_key_rows_venti", "64000000"},
    {"key_generate_method", "RANDOM"},
    {"key_random_engine", "MT19937"},
    {"key_random_shuffle", "1"},
    {"key_random_stride", "1000"},
    {"key_random_stride_short", "10"},
    {"key_random_stride_grande", "100000"},
    {"key_random_stride_venti", "100000"},
}};

template <typename TypeParam>
class OrderByPerfTest : public PerfTest<OrderByConfig>, public testing::Test {
 public:
  template <typename T>
  T GetConfigValue(const char* key, T default_value) const {
    return kConfig.get<T>(key, default_value);
  }
  template <typename T>
  std::vector<T> GetConfigColumn(const char* prefix) const {
    return kConfig.get_column<T>(prefix);
  }
};

template <int N>
struct RadixBits {
  enum {
    VALUE = N,
  };
};

template <typename LaunchParamsAndItemTypeT, typename RadixBitsT>
struct OrderByTestType {
  using launch_params_and_item_type_type = LaunchParamsAndItemTypeT;
  using item_type = typename LaunchParamsAndItemTypeT::item_type;
  using key_type = typename item_type::type;
  using launch_params = typename LaunchParamsAndItemTypeT::launch_params;

  enum {
    BLOCK_THREADS = launch_params::BLOCK_THREADS,
    ITEMS_PER_THREAD = launch_params::ITEMS_PER_THREAD,
    BLOCK_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    RADIX_BITS = RadixBitsT::VALUE,
    NUM_BINS = 1 << RADIX_BITS,
    END_BIT = sizeof(key_type) * /*BITS_PER_BYTE=*/8,
    NUM_PASSES = utils::DivideAndRoundUp<END_BIT, RADIX_BITS>::VALUE,
    HISTOGRAM_SIZE = NUM_BINS * NUM_PASSES,
    BINS_PER_THREAD = utils::DivideAndRoundUp<NUM_BINS, BLOCK_THREADS>::VALUE,
    HISTOGRAM_ITEMS_PER_THREAD = 8,
    HISTOGRAM_TILE_SIZE = 16,
    HISTOGRAM_BLOCK_ITEMS = BLOCK_THREADS * HISTOGRAM_ITEMS_PER_THREAD,
    HISTOGRAM_TILE_ITEMS = HISTOGRAM_TILE_SIZE * HISTOGRAM_BLOCK_ITEMS,
  };

  static std::string GetName() {
    return launch_params_and_item_type_type::GetName() + ".RadixBits" +
           std::to_string(RADIX_BITS);
  }

  static size_t GlobalMemoryLoads(size_t num_keys) {
    int num_histogram_blocks =
        (num_keys + HISTOGRAM_TILE_ITEMS - 1) / HISTOGRAM_TILE_ITEMS;
    // count each atomic add as 1 load + 1 store
    int num_atomic_loads = HISTOGRAM_SIZE * num_histogram_blocks;
    // 1N global memory loads for histogram + 1N for each sorting pass
    return (num_keys * (1ll + NUM_PASSES)) * sizeof(key_type) +
           num_atomic_loads * sizeof(unsigned);
  }

  static size_t GlobalMemoryStores(size_t num_keys) {
    int num_histogram_blocks =
        (num_keys + HISTOGRAM_TILE_ITEMS - 1) / HISTOGRAM_TILE_ITEMS;
    // count the store of each atomic add
    int num_atomic_stores = HISTOGRAM_SIZE * num_histogram_blocks;
    // 1N global memory stores for each sorting pass
    return (num_keys * NUM_PASSES) * sizeof(key_type) +
           num_atomic_stores * sizeof(unsigned);
  }

  template <typename Allocator>
  static void SortKeys(device_column_buffered<key_type>& keys,
                       const Allocator& allocator) {
    using namespace utils;

    // constant size temporary storage that needs to be zero initialized
    struct TempStorage {
      unsigned histogram[HISTOGRAM_SIZE];
      int next_block_idx[NUM_PASSES];
    };

    int num_blocks = (keys.size() + BLOCK_ITEMS - 1) / BLOCK_ITEMS;
    int num_histogram_blocks =
        (keys.size() + HISTOGRAM_TILE_ITEMS - 1) / HISTOGRAM_TILE_ITEMS;

    typedef
        typename Allocator::template rebind<TempStorage>::other temp_alloc_type;
    typedef typename Allocator::template rebind<int>::other int_alloc_type;
    typedef
        typename Allocator::template rebind<unsigned>::other uint_alloc_type;

    auto temp_allocator = temp_alloc_type(allocator);
    auto int_allocator = int_alloc_type(allocator);
    auto uint_allocator = uint_alloc_type(allocator);

    // temporary storage
    device_vector<TempStorage, temp_alloc_type> temp_storage(1, temp_allocator);
    device_vector<unsigned, uint_alloc_type> offsets(HISTOGRAM_SIZE,
                                                     uint_allocator);
    device_vector<unsigned, uint_alloc_type> blocks(
        num_blocks * NUM_BINS * NUM_PASSES, uint_allocator);
    device_vector<int, int_alloc_type> buffer_advancements(NUM_PASSES,
                                                           int_allocator);
    device_vector<int, int_alloc_type> buffer_selectors(2 * NUM_PASSES,
                                                        int_allocator);

    // initialize temporary storage
    cudaMemsetAsync(temp_storage.data(), 0, sizeof(TempStorage));
    cudaMemsetAsync(blocks.data(), 0, sizeof(unsigned) * blocks.size());

    kernels::BuildRadixSortHistogram<BLOCK_THREADS, HISTOGRAM_ITEMS_PER_THREAD,
                                     HISTOGRAM_TILE_SIZE, RADIX_BITS>
        <<<num_histogram_blocks, BLOCK_THREADS>>>(
            keys.ptrs().data(), keys.selector().data(),
            temp_storage.data()->histogram, keys.size());

    // exclusive scan of histogram and set buffer advancements
    kernels::RadixSortHistogramExclusiveScan<BLOCK_THREADS, BINS_PER_THREAD,
                                             RADIX_BITS>
        <<<NUM_PASSES, BLOCK_THREADS>>>(temp_storage.data()->histogram,
                                        offsets.data(),
                                        buffer_advancements.data());

    // update buffer selectors using buffer advancements
    kernels::UpdateBufferSelectors<
        /*BLOCK_THREADS=*/1,
        /*ITEMS_PER_THREAD=*/NUM_PASSES>
        <<</*num_blocks=*/1, /*BLOCK_THREADS=*/1>>>(buffer_advancements.data(),
                                                    keys.selector().data(),
                                                    buffer_selectors.data());

    // start from lsb and loop until no bits are left
    int start_bit = 0;
    for (int pass = 0; pass < NUM_PASSES; ++pass) {
      int num_pass_bits =
          std::min(static_cast<int>(RADIX_BITS), END_BIT - start_bit);

      // input for this sorting pass
      int* pass_next_block_idx = &temp_storage.data()->next_block_idx[pass];
      unsigned* pass_blocks = blocks.data() + pass * num_blocks * NUM_BINS;
      const int* pass_buffer_selectors = buffer_selectors.data() + pass * 2;
      const unsigned* pass_offsets = offsets.data() + pass * NUM_BINS;

      // radix sorting pass
      kernels::RadixSort<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS>
          <<<num_blocks, BLOCK_THREADS>>>(
              pass_buffer_selectors, pass_offsets, start_bit, num_pass_bits,
              keys.ptrs().data(), pass_next_block_idx, pass_blocks,
              keys.size());

      // advance start bit for next pass
      start_bit += RADIX_BITS;
    }
  }
};

using LaunchParamsTypes =
    std::tuple<LaunchParams<256, 8>, LaunchParams<256, 16>,
               LaunchParams<256, 24>>;

using LaunchParamsAndItemTypes =
    CombineLaunchParamsAndTypes<LaunchParamsAndItemType, LaunchParamsTypes, int,
                                unsigned, long long, unsigned long long>;

using TestTypes =
    MakeTestTypes<CombineTestTypes<OrderByTestType, LaunchParamsAndItemTypes,
                                   RadixBits<8>>>::types;

TYPED_TEST_SUITE(OrderByPerfTest, TestTypes, TestTypeNames);

TYPED_TEST(OrderByPerfTest, SelectKeysOrderByKeys) {
  using key_type = typename TypeParam::key_type;
  using indices_type = utils::size_type;

  auto items = this->template GetConfigColumn<key_type>("key");
  ASSERT_NE(items.size(), 0u);

  auto check_result = this->GetConfigValue("check_result", true);
  auto result_file = this->GetConfigValue("result_file", std::string());

  device_column_buffered<key_type> d_items(items.size());
  int input_selector = 0;
  d_items.buffer(input_selector).copy_from_host(items.data(), items.size());
  d_items.selector().copy_from_host(&input_selector, 1);

  auto free_list = std::make_shared<
      caching_device_allocator<utils::size_type>::free_list_type>();
  caching_device_allocator<utils::size_type> allocator(free_list);

  // provide throughput information
  this->set_element_count(items.size());
  this->set_element_size(sizeof(key_type));
  this->set_elements_per_thread(TypeParam::launch_params::ITEMS_PER_THREAD);
  this->set_global_memory_loads(TypeParam::GlobalMemoryLoads(items.size()));
  this->set_global_memory_stores(TypeParam::GlobalMemoryStores(items.size()));

  this->Measure(kConfig, [&]() { TypeParam::SortKeys(d_items, allocator); });

  if (check_result) {
    int output_selector;
    d_items.selector().copy_to_host(&output_selector, 1);
    std::vector<key_type> h_sorted_items(items.size());
    d_items.buffer(output_selector)
        .copy_to_host(h_sorted_items.data(), items.size());
    std::vector<key_type> expected_sorted_items = items;
    std::stable_sort(expected_sorted_items.begin(),
                     expected_sorted_items.end());
    EXPECT_EQ(h_sorted_items, expected_sorted_items);
  }

  if (!result_file.empty()) {
    int output_selector;
    d_items.selector().copy_to_host(&output_selector, 1);
    std::vector<key_type> h_sorted_items(items.size());
    d_items.buffer(output_selector)
        .copy_to_host(h_sorted_items.data(), items.size());

    std::ofstream result_out;
    result_out.open(result_file);
    ASSERT_TRUE(result_out.is_open())
        << "failed to open result file: " << result_file;
    result_out << "sorted_item" << std::endl;
    for (size_t i = 0; i < h_sorted_items.size(); ++i) {
      result_out << h_sorted_items[i] << std::endl;
    }
    result_out.close();
  }

  for (auto entry : *free_list) {
    cudaFree(entry.second);
  }
}

}  // namespace test
}  // namespace breeze
