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

// Microbenchmark: the ROO preproc's scatter_add_ into a [256] int64 array.
//
// The graph computes, per batch row, the sum of a long value over that row's
// elements:
//   index = repeat_interleave(arange-like rows, lengths)   sorted, in [0, 256)
//   out   = zeros([256], long).scatter_add_(0, index, src)
// TorchWave fuses it as tw.scatter_add (tw_scatter_add in Elementwise.cuh):
// global 64-bit atomics, warp-aggregated when all active lanes of a warp hit
// one destination. The index is sorted, so long runs of consecutive elements
// add into the same address.
//
// Variants, each a grid-strided loop over the elements:
//   globalAtomic   atomicAdd(&out[index[i]], src[i]) in global memory. With
//                  "agg" this is what tw.scatter_add does.
//   sharedBarrier  each block accumulates into its own [256] array in shared
//                  memory, writes it to a per-block slice of a global partials
//                  buffer, and passes an opBarrier (Core.cuh). Then each block
//                  sums one bin over all blocks' partials into out, its
//                  threads splitting the partials.
//   sharedAtomic   like sharedBarrier, but each block adds its shared array
//                  into out with one global atomicAdd per nonzero bin, with no
//                  barrier.
// Each runs plain and with warp aggregation ("agg"): a warp whose lanes all
// hit one bin sums them with shuffles and does one atomic instead of 32.
//
// Data comes from one of:
//   --data          operands written by --capture (from a reference frame saved
//                   by executor_test --save_reference_frame) or by
//                   torchwave_meta/tests:capture_scatter_add;
//   --lengths_file  the per-row repeat counts of the repeat_interleave that
//                   builds the index, as traced from the graph, with a 0/1 src;
//   neither         synthetic lengths of the same shape.
// --shuffle puts the elements in random order, for the random-index case.

#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>

#include "velox/experimental/torchwave/Core.cuh"

namespace torch::wave {
// From Utils.h, which cannot be included next to Core.cuh: both define
// roundUp.
std::unordered_map<int32_t, c10::IValue> loadReferenceFrame(
    const std::string& path);
} // namespace torch::wave

DEFINE_string(
    data,
    "",
    "Data file written by --capture. When empty, runs on synthetic data.");
DEFINE_bool(
    capture,
    false,
    "Read --frame and write the scatter_add operands to --data, then exit.");
DEFINE_string(
    frame,
    "",
    "Reference frame from executor_test --save_reference_frame.");
DEFINE_int32(index_id, -1, "Value id of the scatter_add index in --frame.");
DEFINE_int32(src_id, -1, "Value id of the scatter_add src in --frame.");
DEFINE_int32(
    out_id,
    -1,
    "Value id of the scatter_add result in --frame, used as the expected "
    "output. Optional; the CPU sum is used when absent.");
DEFINE_string(
    lengths_file,
    "",
    "Text file of per-row element counts (the repeat counts of the "
    "repeat_interleave that builds the index), one per line. The index is "
    "each row's number repeated that many times, as in the graph.");
DEFINE_double(
    src_fraction,
    1.0,
    "Fraction of src elements that are 1 rather than 0, for --lengths_file "
    "and synthetic data. ROO's src is a 0/1 mask.");
DEFINE_bool(
    shuffle,
    false,
    "Randomly permute the elements (index and src together), keeping every "
    "bin's total. Models a scatter_add with random rather than sorted "
    "indices, where warp aggregation finds no warps with a single bin.");
DEFINE_int32(synthetic_rows, 256, "Bins (batch rows) for synthetic data.");
DEFINE_int32(
    synthetic_mean_length,
    50,
    "Mean elements per row for synthetic data; lengths are uniform in "
    "[0, 2 * mean].");
DEFINE_int32(block_size, 256, "Threads per block.");
DEFINE_int32(
    num_blocks,
    0,
    "Blocks per launch. 0 runs two grids: one block per block_size elements "
    "(how tw.scatter_add launches), and every block resident at once (the "
    "most the barrier variant can use).");
DEFINE_int32(iterations, 200, "Timed launches per variant.");

namespace {

using torch::wave::BlockInfo;

constexpr uint32_t kMagic = 0x44444153; // "SADD"
constexpr int32_t kMaxBins = 1024;

#define CUDA_CHECK(expr)              \
  do {                                \
    cudaError_t err = (expr);         \
    if (err != cudaSuccess) {         \
      fprintf(                        \
          stderr,                     \
          "CUDA error %s at %s:%d\n", \
          cudaGetErrorString(err),    \
          __FILE__,                   \
          __LINE__);                  \
      exit(1);                        \
    }                                 \
  } while (0)

// The operands of one scatter_add_ into a 1-d int64 array.
struct ScatterData {
  int64_t numBins{0};
  std::vector<int64_t> index;
  std::vector<int64_t> src;
  std::vector<int64_t> expected;
};

void writeData(const ScatterData& data, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  const int64_t numElements = static_cast<int64_t>(data.index.size());
  out.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
  out.write(reinterpret_cast<const char*>(&numElements), sizeof(int64_t));
  out.write(reinterpret_cast<const char*>(&data.numBins), sizeof(int64_t));
  out.write(
      reinterpret_cast<const char*>(data.index.data()),
      numElements * sizeof(int64_t));
  out.write(
      reinterpret_cast<const char*>(data.src.data()),
      numElements * sizeof(int64_t));
  out.write(
      reinterpret_cast<const char*>(data.expected.data()),
      data.numBins * sizeof(int64_t));
  if (!out.good()) {
    fprintf(stderr, "Cannot write %s\n", path.c_str());
    exit(1);
  }
}

ScatterData readData(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  uint32_t magic = 0;
  int64_t numElements = 0;
  ScatterData data;
  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char*>(&numElements), sizeof(int64_t));
  in.read(reinterpret_cast<char*>(&data.numBins), sizeof(int64_t));
  if (!in.good() || magic != kMagic || data.numBins <= 0 ||
      data.numBins > kMaxBins) {
    fprintf(stderr, "Bad scatter_add data file: %s\n", path.c_str());
    exit(1);
  }
  data.index.resize(numElements);
  data.src.resize(numElements);
  data.expected.resize(data.numBins);
  in.read(
      reinterpret_cast<char*>(data.index.data()),
      numElements * sizeof(int64_t));
  in.read(
      reinterpret_cast<char*>(data.src.data()), numElements * sizeof(int64_t));
  in.read(
      reinterpret_cast<char*>(data.expected.data()),
      data.numBins * sizeof(int64_t));
  if (!in.good()) {
    fprintf(stderr, "Truncated scatter_add data file: %s\n", path.c_str());
    exit(1);
  }
  return data;
}

std::vector<int64_t> cpuScatterAdd(
    const std::vector<int64_t>& index,
    const std::vector<int64_t>& src,
    int64_t numBins) {
  std::vector<int64_t> out(numBins, 0);
  for (size_t i = 0; i < index.size(); ++i) {
    out[index[i]] += src[i];
  }
  return out;
}

// The int64 contents of 'value', which must be a 1-d integer tensor.
std::vector<int64_t> int64Contents(const c10::IValue& value, int32_t id) {
  if (!value.isTensor() || value.toTensor().dim() != 1) {
    fprintf(stderr, "Value %d is not a 1-d tensor\n", id);
    exit(1);
  }
  const auto tensor = value.toTensor().to(at::kCPU).to(at::kLong).contiguous();
  const auto* begin = tensor.data_ptr<int64_t>();
  return std::vector<int64_t>(begin, begin + tensor.numel());
}

int capture() {
  if (FLAGS_frame.empty() || FLAGS_data.empty() || FLAGS_index_id < 0 ||
      FLAGS_src_id < 0) {
    fprintf(
        stderr, "--capture needs --frame, --data, --index_id and --src_id\n");
    return 1;
  }
  const auto frame = torch::wave::loadReferenceFrame(FLAGS_frame);
  auto lookup = [&](int32_t id) -> const c10::IValue& {
    auto it = frame.find(id);
    if (it == frame.end()) {
      fprintf(stderr, "Value %d is not in %s\n", id, FLAGS_frame.c_str());
      exit(1);
    }
    return it->second;
  };
  ScatterData data;
  data.index = int64Contents(lookup(FLAGS_index_id), FLAGS_index_id);
  data.src = int64Contents(lookup(FLAGS_src_id), FLAGS_src_id);
  if (data.index.size() != data.src.size()) {
    fprintf(
        stderr,
        "index has %zu elements, src %zu\n",
        data.index.size(),
        data.src.size());
    return 1;
  }
  if (FLAGS_out_id >= 0) {
    data.expected = int64Contents(lookup(FLAGS_out_id), FLAGS_out_id);
    data.numBins = static_cast<int64_t>(data.expected.size());
  } else {
    data.numBins = data.index.empty()
        ? 0
        : *std::max_element(data.index.begin(), data.index.end()) + 1;
    data.expected = cpuScatterAdd(data.index, data.src, data.numBins);
  }
  for (auto bin : data.index) {
    if (bin < 0 || bin >= data.numBins) {
      fprintf(stderr, "Index %ld out of [0, %ld)\n", bin, data.numBins);
      return 1;
    }
  }
  writeData(data, FLAGS_data);
  printf(
      "Wrote %s: %zu elements, %ld bins\n",
      FLAGS_data.c_str(),
      data.index.size(),
      data.numBins);
  return 0;
}

// Per-row element counts: from --lengths_file, else uniform in
// [0, 2 * synthetic_mean_length].
std::vector<int64_t> rowLengths(std::mt19937_64& rng) {
  std::vector<int64_t> lengths;
  if (!FLAGS_lengths_file.empty()) {
    std::ifstream in(FLAGS_lengths_file);
    int64_t length = 0;
    while (in >> length) {
      lengths.push_back(length);
    }
    if (lengths.empty()) {
      fprintf(stderr, "No lengths in %s\n", FLAGS_lengths_file.c_str());
      exit(1);
    }
    return lengths;
  }
  std::uniform_int_distribution<int32_t> lengthDist(
      0, 2 * FLAGS_synthetic_mean_length);
  for (int32_t row = 0; row < FLAGS_synthetic_rows; ++row) {
    lengths.push_back(lengthDist(rng));
  }
  return lengths;
}

ScatterData generatedData() {
  std::mt19937_64 rng(1);
  std::bernoulli_distribution isOne(FLAGS_src_fraction);
  const auto lengths = rowLengths(rng);
  ScatterData data;
  data.numBins = static_cast<int64_t>(lengths.size());
  for (int64_t row = 0; row < data.numBins; ++row) {
    for (int64_t i = 0; i < lengths[row]; ++i) {
      data.index.push_back(row);
      data.src.push_back(isOne(rng) ? 1 : 0);
    }
  }
  data.expected = cpuScatterAdd(data.index, data.src, data.numBins);
  return data;
}

__device__ inline void atomicAddLong(int64_t* address, int64_t value) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(value));
}

constexpr uint32_t kFullMask = 0xffffffffu;

// Adds element i into 'target' at index[i], for every lane of the warp, which
// must be converged; lanes past the end pass active = false. With
// kWarpAggregate, a warp whose active lanes all hit one bin sums them with
// shuffles and adds once from its first active lane. The index is sorted, so
// that is most warps.
template <bool kWarpAggregate>
__device__ inline void
addElement(int64_t* target, bool active, int64_t key, int64_t value) {
  if constexpr (kWarpAggregate) {
    const uint32_t activeMask = __ballot_sync(kFullMask, active);
    const int32_t leader = __ffs(activeMask) - 1;
    const int64_t leaderKey = __shfl_sync(kFullMask, key, leader);
    if (__all_sync(kFullMask, !active || key == leaderKey)) {
      int64_t sum = active ? value : 0;
      for (int32_t offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(kFullMask, sum, offset);
      }
      if ((threadIdx.x & 31) == leader) {
        atomicAddLong(&target[leaderKey], sum);
      }
      return;
    }
  }
  if (active) {
    atomicAddLong(&target[key], value);
  }
}

// Calls addElement for this block's share of the elements. Each warp runs
// until all of its lanes are past the end, so warp collectives stay
// converged.
template <bool kWarpAggregate>
__device__ inline void accumulate(
    const int64_t* index,
    const int64_t* src,
    int64_t numElements,
    int64_t* target) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;;
       i += stride) {
    const bool active = i < numElements;
    if (!__any_sync(kFullMask, active)) {
      break;
    }
    addElement<kWarpAggregate>(
        target, active, active ? index[i] : 0, active ? src[i] : 0);
  }
}

template <bool kWarpAggregate>
__global__ void globalAtomicKernel(
    const int64_t* index,
    const int64_t* src,
    int64_t numElements,
    int64_t* out) {
  accumulate<kWarpAggregate>(index, src, numElements, out);
}

// Accumulates this block's share of the elements into 'bins' in shared memory.
template <bool kWarpAggregate>
__device__ void accumulateShared(
    const int64_t* index,
    const int64_t* src,
    int64_t numElements,
    int32_t numBins,
    int64_t* bins) {
  for (auto bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
    bins[bin] = 0;
  }
  __syncthreads();
  accumulate<kWarpAggregate>(index, src, numElements, bins);
  __syncthreads();
}

template <bool kWarpAggregate>
__global__ void sharedBarrierKernel(
    const int64_t* index,
    const int64_t* src,
    int64_t numElements,
    int32_t numBins,
    int64_t* partials,
    int32_t* barrierCounter,
    int64_t* out) {
  __shared__ int64_t bins[kMaxBins];
  __shared__ BlockInfo block;
  if (threadIdx.x == 0) {
    block.blockInOp = blockIdx.x;
    block.numBlocksInOp = gridDim.x;
    block.params = barrierCounter;
    block.barrierClocks = 0;
  }
  accumulateShared<kWarpAggregate>(index, src, numElements, numBins, bins);
  for (auto bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
    partials[blockIdx.x * static_cast<int64_t>(numBins) + bin] = bins[bin];
  }
  torch::wave::opBarrier(block, 0);
  // Each block sums one bin over all blocks' partials: its threads split the
  // partials, then reduce in shared memory, reusing 'bins'.
  for (auto bin = blockIdx.x; bin < numBins; bin += gridDim.x) {
    int64_t sum = 0;
    for (auto b = threadIdx.x; b < gridDim.x; b += blockDim.x) {
      sum += partials[b * static_cast<int64_t>(numBins) + bin];
    }
    __syncthreads();
    bins[threadIdx.x] = sum;
    __syncthreads();
    for (auto width = blockDim.x / 2; width > 0; width >>= 1) {
      if (threadIdx.x < width) {
        bins[threadIdx.x] += bins[threadIdx.x + width];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      out[bin] = bins[0];
    }
  }
}

template <bool kWarpAggregate>
__global__ void sharedAtomicKernel(
    const int64_t* index,
    const int64_t* src,
    int64_t numElements,
    int32_t numBins,
    int64_t* out) {
  __shared__ int64_t bins[kMaxBins];
  accumulateShared<kWarpAggregate>(index, src, numElements, numBins, bins);
  for (auto bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
    if (bins[bin] != 0) {
      atomicAddLong(&out[bin], bins[bin]);
    }
  }
}

enum class Variant { kGlobalAtomic, kSharedBarrier, kSharedAtomic };

const char* variantName(Variant variant) {
  switch (variant) {
    case Variant::kGlobalAtomic:
      return "globalAtomic";
    case Variant::kSharedBarrier:
      return "sharedBarrier";
    case Variant::kSharedAtomic:
      return "sharedAtomic";
  }
  return "?";
}

struct DeviceData {
  int64_t* index{nullptr};
  int64_t* src{nullptr};
  int64_t* out{nullptr};
  int64_t* partials{nullptr};
  int32_t* barrierCounter{nullptr};
  int64_t numElements{0};
  int32_t numBins{0};
};

// Zeroes the output (and the barrier counter, used once per launch) and
// launches 'variant' once.
template <bool kWarpAggregate>
void launchVariant(Variant variant, const DeviceData& d, int32_t numBlocks) {
  CUDA_CHECK(cudaMemsetAsync(d.out, 0, d.numBins * sizeof(int64_t)));
  switch (variant) {
    case Variant::kGlobalAtomic:
      globalAtomicKernel<kWarpAggregate><<<numBlocks, FLAGS_block_size>>>(
          d.index, d.src, d.numElements, d.out);
      break;
    case Variant::kSharedBarrier: {
      CUDA_CHECK(cudaMemsetAsync(d.barrierCounter, 0, sizeof(int32_t)));
      // opBarrier waits for every block, so all of them must be resident.
      void* args[] = {
          const_cast<int64_t**>(&d.index),
          const_cast<int64_t**>(&d.src),
          const_cast<int64_t*>(&d.numElements),
          const_cast<int32_t*>(&d.numBins),
          const_cast<int64_t**>(&d.partials),
          const_cast<int32_t**>(&d.barrierCounter),
          const_cast<int64_t**>(&d.out)};
      CUDA_CHECK(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(sharedBarrierKernel<kWarpAggregate>),
          numBlocks,
          FLAGS_block_size,
          args));
      break;
    }
    case Variant::kSharedAtomic:
      sharedAtomicKernel<kWarpAggregate><<<numBlocks, FLAGS_block_size>>>(
          d.index, d.src, d.numElements, d.numBins, d.out);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

void launch(
    Variant variant,
    bool warpAggregate,
    const DeviceData& d,
    int32_t numBlocks) {
  if (warpAggregate) {
    launchVariant<true>(variant, d, numBlocks);
  } else {
    launchVariant<false>(variant, d, numBlocks);
  }
}

// Microseconds per launch, including the output memset; checks the result.
double timeVariant(
    Variant variant,
    bool warpAggregate,
    const DeviceData& d,
    int32_t numBlocks,
    const std::vector<int64_t>& expected,
    bool& correct) {
  launch(variant, warpAggregate, d, numBlocks);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int64_t> result(d.numBins);
  CUDA_CHECK(cudaMemcpy(
      result.data(),
      d.out,
      d.numBins * sizeof(int64_t),
      cudaMemcpyDeviceToHost));
  correct = result == expected;
  for (int32_t i = 0; i < 3; ++i) {
    launch(variant, warpAggregate, d, numBlocks);
  }
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int32_t i = 0; i < FLAGS_iterations; ++i) {
    launch(variant, warpAggregate, d, numBlocks);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 1000.0 * ms / FLAGS_iterations;
}

// Microseconds per output memset alone, the floor every variant includes.
double timeMemsetOnly(const DeviceData& d) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int32_t i = 0; i < FLAGS_iterations; ++i) {
    CUDA_CHECK(cudaMemsetAsync(d.out, 0, d.numBins * sizeof(int64_t)));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 1000.0 * ms / FLAGS_iterations;
}

void printShape(const ScatterData& data) {
  int64_t longestRun = 0;
  int64_t run = 0;
  int64_t usedBins = 0;
  for (size_t i = 0; i < data.index.size(); ++i) {
    run = (i > 0 && data.index[i] == data.index[i - 1]) ? run + 1 : 1;
    longestRun = std::max(longestRun, run);
  }
  for (auto value : data.expected) {
    usedBins += value != 0;
  }
  const bool sorted = std::is_sorted(data.index.begin(), data.index.end());
  printf(
      "%zu elements, %ld bins (%ld nonzero), index %s, longest run of one "
      "bin %ld\n",
      data.index.size(),
      data.numBins,
      usedBins,
      sorted ? "sorted" : "unsorted",
      longestRun);
}

// Applies one random permutation to index and src.
void shuffleElements(ScatterData& data) {
  std::mt19937_64 rng(2);
  for (size_t i = data.index.size(); i > 1; --i) {
    const size_t j = std::uniform_int_distribution<size_t>(0, i - 1)(rng);
    std::swap(data.index[i - 1], data.index[j]);
    std::swap(data.src[i - 1], data.src[j]);
  }
}

int run() {
  auto data = FLAGS_data.empty() ? generatedData() : readData(FLAGS_data);
  if (FLAGS_shuffle) {
    shuffleElements(data);
    printf("Elements shuffled into random order\n");
  }
  if (!FLAGS_data.empty()) {
    printf("Data: %s\n", FLAGS_data.c_str());
  } else {
    printf(
        "Data: index from %s, src 1 with probability %.2f\n",
        FLAGS_lengths_file.empty() ? "synthetic lengths"
                                   : FLAGS_lengths_file.c_str(),
        FLAGS_src_fraction);
  }
  printShape(data);
  if (data.numBins > kMaxBins) {
    fprintf(stderr, "More than %d bins\n", kMaxBins);
    return 1;
  }
  // sharedBarrier's final reduction halves the block in shared memory.
  if (FLAGS_block_size < 32 || FLAGS_block_size > kMaxBins ||
      (FLAGS_block_size & (FLAGS_block_size - 1)) != 0) {
    fprintf(
        stderr, "--block_size must be a power of two in [32, %d]\n", kMaxBins);
    return 1;
  }
  if (cpuScatterAdd(data.index, data.src, data.numBins) != data.expected) {
    printf(
        "Note: the captured output differs from the CPU sum of the inputs\n");
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int32_t barrierBlocksPerSm = 0;
  int32_t aggregatedBlocksPerSm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &barrierBlocksPerSm, sharedBarrierKernel<false>, FLAGS_block_size, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &aggregatedBlocksPerSm, sharedBarrierKernel<true>, FLAGS_block_size, 0));
  barrierBlocksPerSm = std::min(barrierBlocksPerSm, aggregatedBlocksPerSm);
  const int32_t maxResident = barrierBlocksPerSm * prop.multiProcessorCount;
  const auto numElements = static_cast<int64_t>(data.index.size());
  const auto perElementBlocks = static_cast<int32_t>(std::max<int64_t>(
      1, (numElements + FLAGS_block_size - 1) / FLAGS_block_size));

  std::vector<int32_t> grids;
  if (FLAGS_num_blocks > 0) {
    grids.push_back(FLAGS_num_blocks);
  } else {
    grids.push_back(perElementBlocks);
    if (maxResident != perElementBlocks) {
      grids.push_back(maxResident);
    }
  }
  const int32_t maxGrid = *std::max_element(grids.begin(), grids.end());

  DeviceData d;
  d.numElements = numElements;
  d.numBins = static_cast<int32_t>(data.numBins);
  const size_t bytes = std::max<int64_t>(numElements, 1) * sizeof(int64_t);
  CUDA_CHECK(cudaMalloc(&d.index, bytes));
  CUDA_CHECK(cudaMalloc(&d.src, bytes));
  CUDA_CHECK(cudaMalloc(&d.out, data.numBins * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(
      &d.partials,
      static_cast<size_t>(maxGrid) * data.numBins * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d.barrierCounter, sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(
      d.index,
      data.index.data(),
      numElements * sizeof(int64_t),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      d.src,
      data.src.data(),
      numElements * sizeof(int64_t),
      cudaMemcpyHostToDevice));

  printf(
      "%s, %d SMs, block %d, at most %d resident barrier blocks, %d timed "
      "launches\n",
      prop.name,
      prop.multiProcessorCount,
      FLAGS_block_size,
      maxResident,
      FLAGS_iterations);
  printf(
      "Times include a %ld-byte output memset (%.2f us alone).\n",
      data.numBins * static_cast<int64_t>(sizeof(int64_t)),
      timeMemsetOnly(d));
  printf(
      "%-14s %-5s %8s %10s %8s\n",
      "variant",
      "warp",
      "blocks",
      "us/launch",
      "result");
  for (auto numBlocks : grids) {
    for (auto variant :
         {Variant::kGlobalAtomic,
          Variant::kSharedBarrier,
          Variant::kSharedAtomic}) {
      for (bool warpAggregate : {false, true}) {
        const char* warp = warpAggregate ? "agg" : "-";
        if (variant == Variant::kSharedBarrier && numBlocks > maxResident) {
          printf(
              "%-14s %-5s %8d %10s %8s\n",
              variantName(variant),
              warp,
              numBlocks,
              "-",
              "skipped: more blocks than can be resident");
          continue;
        }
        bool correct = false;
        const double us = timeVariant(
            variant, warpAggregate, d, numBlocks, data.expected, correct);
        printf(
            "%-14s %-5s %8d %10.2f %8s\n",
            variantName(variant),
            warp,
            numBlocks,
            us,
            correct ? "ok" : "WRONG");
      }
    }
  }
  CUDA_CHECK(cudaFree(d.index));
  CUDA_CHECK(cudaFree(d.src));
  CUDA_CHECK(cudaFree(d.out));
  CUDA_CHECK(cudaFree(d.partials));
  CUDA_CHECK(cudaFree(d.barrierCounter));
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return FLAGS_capture ? capture() : run();
}
