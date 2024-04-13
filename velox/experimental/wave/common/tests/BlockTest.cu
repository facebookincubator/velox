#include "velox/experimental/wave/common/Block.cuh"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/tests/BlockTest.h"

namespace facebook::velox::wave {

using ScanAlgorithm = cub::BlockScan<int, 256, cub::BLOCK_SCAN_RAKING>;

__global__ void boolToIndices(
    uint8_t** bools,
    int32_t** indices,
    int32_t* sizes,
    int64_t* times) {
  extern __shared__ __align__(alignof(ScanAlgorithm::TempStorage)) char smem[];
  int32_t idx = blockIdx.x;
  // Start cycle timer
  clock_t start = clock();
  uint8_t* blockBools = bools[idx];
  boolBlockToIndices<256>(
      [&]() { return blockBools[threadIdx.x]; },
      idx * 256,
      indices[idx],
      smem,
      sizes[idx]);
  clock_t stop = clock();
  if (threadIdx.x == 0) {
    times[idx] = (start > stop) ? start - stop : stop - start;
  }
}

void BlockTestStream::testBoolToIndices(
    int32_t numBlocks,
    uint8_t** flags,
    int32_t** indices,
    int32_t* sizes,
    int64_t* times) {
  CUDA_CHECK(cudaGetLastError());
  auto tempBytes = sizeof(typename ScanAlgorithm::TempStorage);
  boolToIndices<<<numBlocks, 256, tempBytes, stream_->stream>>>(
      flags, indices, sizes, times);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void boolToIndicesNoShared(
    uint8_t** bools,
    int32_t** indices,
    int32_t* sizes,
    int64_t* times,
    void* temp) {
  int32_t idx = blockIdx.x;

  // Start cycle timer
  clock_t start = clock();
  uint8_t* blockBools = bools[idx];
  char* smem = reinterpret_cast<char*>(temp) +
      blockIdx.x * sizeof(typename ScanAlgorithm::TempStorage);
  boolBlockToIndices<256>(
      [&]() { return blockBools[threadIdx.x]; },
      idx * 256,
      indices[idx],
      smem,
      sizes[idx]);
  clock_t stop = clock();
  if (threadIdx.x == 0) {
    times[idx] = (start > stop) ? start - stop : stop - start;
  }
}

void BlockTestStream::testBoolToIndicesNoShared(
    int32_t numBlocks,
    uint8_t** flags,
    int32_t** indices,
    int32_t* sizes,
    int64_t* times,
    void* temp) {
  CUDA_CHECK(cudaGetLastError());
  boolToIndicesNoShared<<<numBlocks, 256, 0, stream_->stream>>>(
      flags, indices, sizes, times, temp);
  CUDA_CHECK(cudaGetLastError());
}

int32_t BlockTestStream::boolToIndicesSize() {
  return sizeof(typename ScanAlgorithm::TempStorage);
}

__global__ void sum64(int64_t* numbers, int64_t* results) {
  extern __shared__ __align__(
      alignof(cub::BlockReduce<int64_t, 256>::TempStorage)) char smem[];
  int32_t idx = blockIdx.x;
  blockSum<256>(
      [&]() { return numbers[idx * 256 + threadIdx.x]; }, smem, results);
}

void BlockTestStream::testSum64(
    int32_t numBlocks,
    int64_t* numbers,
    int64_t* results) {
  auto tempBytes = sizeof(typename cub::BlockReduce<int64_t, 256>::TempStorage);
  sum64<<<numBlocks, 256, tempBytes, stream_->stream>>>(numbers, results);
  CUDA_CHECK(cudaGetLastError());
}

/// Keys and values are n sections of 8K items. The items in each section get
/// sorted on the key.
void __global__ __launch_bounds__(1024)
    testSort(uint16_t** keys, uint16_t** values) {
  extern __shared__ __align__(16) char smem[];
  auto keyBase = keys[blockIdx.x];
  auto valueBase = values[blockIdx.x];
  blockSort<1024, 8>(
      [&](auto i) { return keyBase[i]; },
      [&](auto i) { return valueBase[i]; },
      keys[blockIdx.x],
      values[blockIdx.x],
      smem);
}

void BlockTestStream::testSort16(
    int32_t numBlocks,
    uint16_t** keys,
    uint16_t** values) {
  auto tempBytes = sizeof(
      typename cub::BlockRadixSort<uint16_t, 1024, 8, uint16_t>::TempStorage);

  testSort<<<1024, numBlocks, tempBytes, stream_->stream>>>(keys, values);
}

REGISTER_KERNEL("testSort", testSort);
REGISTER_KERNEL("boolToIndices", boolToIndices);
REGISTER_KERNEL("sum64", sum64);

} // namespace facebook::velox::wave
