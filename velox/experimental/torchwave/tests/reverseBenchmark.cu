// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Microbenchmark: does coalesced global memory run as fast descending as
// ascending?
//
// A 4 GB int64 array is filled with src[i] = i and copied in reverse
// (dst[i] = src[n-1-i]) three ways, each a grid-strided loop launched with an
// occupancy-full grid (numSMs * maxActiveBlocksPerSM):
//   forwardCopy   dst[i] = src[i]                 -- baseline, both ascending.
//   reverseDirect dst[n-1-i] = src[i]             -- ascending read, DESCENDING
//                                                    write.
//   reverseShared read a block chunk ascending into shared, then write the
//                 mirrored chunk with an ascending global write whose values
//                 come from a reversed shared read -- both global accesses
//                 ascending, the reversal is confined to shared memory.
//
// If descending coalesced access is as fast as ascending, reverseDirect matches
// forwardCopy and there is nothing to gain from reverseShared.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kBlockSize = 256;
constexpr int kIters = 10;

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
      std::abort();                   \
    }                                 \
  } while (0)

__global__ void fillIota(int64_t* src, int64_t n) {
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += stride) {
    src[i] = i;
  }
}

__global__ void forwardCopy(const int64_t* src, int64_t* dst, int64_t n) {
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += stride) {
    dst[i] = src[i];
  }
}

__global__ void reverseDirect(const int64_t* src, int64_t* dst, int64_t n) {
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += stride) {
    // Ascending read, descending write.
    dst[n - 1 - i] = src[i];
  }
}

// One block-sized chunk of src is read ascending into shared, then written to
// its mirrored position in dst with an ascending global write whose value comes
// from a reversed shared read. Both global accesses are ascending. Requires
// n % blockDim.x == 0.
__global__ void reverseShared(const int64_t* src, int64_t* dst, int64_t n) {
  __shared__ int64_t tile[kBlockSize];
  int64_t numChunks = n / blockDim.x;
  for (int64_t chunk = blockIdx.x; chunk < numChunks; chunk += gridDim.x) {
    int64_t base = chunk * blockDim.x;
    tile[threadIdx.x] = src[base + threadIdx.x];
    __syncthreads();
    // The mirror of the source chunk [base, base + B) is dst [n-base-B,
    // n-base).
    int64_t dstBase = n - base - blockDim.x;
    dst[dstBase + threadIdx.x] = tile[blockDim.x - 1 - threadIdx.x];
    __syncthreads();
  }
}

__global__ void
countErrors(const int64_t* dst, int64_t n, bool reversed, int* errors) {
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += stride) {
    int64_t expected = reversed ? (n - 1 - i) : i;
    if (dst[i] != expected) {
      atomicAdd(errors, 1);
    }
  }
}

int occupancyGrid(const void* kernel, size_t sharedBytes, int numSms) {
  int blocksPerSm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSm, kernel, kBlockSize, sharedBytes));
  return numSms * blocksPerSm;
}

// Times 'kIters' launches of 'launch' with CUDA events and returns the average
// milliseconds per launch.
template <typename Launch>
float timeMs(Launch launch) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  launch(); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kIters; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / kIters;
}

int64_t verify(const int64_t* dst, int64_t n, bool reversed, int numSms) {
  int* dErrors = nullptr;
  CUDA_CHECK(cudaMalloc(&dErrors, sizeof(int)));
  CUDA_CHECK(cudaMemset(dErrors, 0, sizeof(int)));
  countErrors<<<numSms * 32, kBlockSize>>>(dst, n, reversed, dErrors);
  CUDA_CHECK(cudaDeviceSynchronize());
  int hErrors = 0;
  CUDA_CHECK(
      cudaMemcpy(&hErrors, dErrors, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dErrors));
  return hErrors;
}

} // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int numSms = prop.multiProcessorCount;
  // Nominal (theoretical peak) HBM bandwidth: 2 (DDR) * clock(Hz) * bytes/txn.
  double nominalGbps = 2.0 * static_cast<double>(prop.memoryClockRate) * 1.0e3 *
      (prop.memoryBusWidth / 8) / 1.0e9;

  const int64_t kArrayBytes = 4LL * 1024 * 1024 * 1024; // 4 GiB per array.
  int64_t n = kArrayBytes / static_cast<int64_t>(sizeof(int64_t));
  n = (n / kBlockSize) * kBlockSize;
  double gib =
      static_cast<double>(n * sizeof(int64_t)) / (1024.0 * 1024.0 * 1024.0);
  // Each copy reads the whole array and writes the whole array.
  double bytesPerCopy = 2.0 * static_cast<double>(n) * sizeof(int64_t);

  printf(
      "GPU: %s, %d SMs. Array: %ld int64 (%.2f GiB), copy moves %.2f GiB "
      "(read+write). %d iters/kernel. Nominal BW: %.1f GB/s.\n",
      prop.name,
      numSms,
      static_cast<long>(n),
      gib,
      2 * gib,
      kIters,
      nominalGbps);

  int64_t* src = nullptr;
  int64_t* dst = nullptr;
  CUDA_CHECK(cudaMalloc(&src, n * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&dst, n * sizeof(int64_t)));

  int gridFill = occupancyGrid((const void*)fillIota, 0, numSms);
  fillIota<<<gridFill, kBlockSize>>>(src, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // For each kernel, sweep achieved occupancy by launching one wave at a
  // fraction of the max resident blocks/SM (grid = numSms * round(frac *
  // blocksPerSm)). This models the effect register pressure would have: fewer
  // resident blocks/SM. A memory-bound copy that already saturates below 100%
  // occupancy will hold its throughput as the fraction drops.
  auto run =
      [&](const char* name, auto kernel, size_t sharedBytes, bool reversed) {
        int blocksPerSm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, (const void*)kernel, kBlockSize, sharedBytes));
        printf(
            "  %-14s max occupancy: %d blocks/SM (%d threads/SM)\n",
            name,
            blocksPerSm,
            blocksPerSm * kBlockSize);
        const double fracs[] = {1.0, 0.75, 0.5};
        for (double frac : fracs) {
          int bps = static_cast<int>(llround(frac * blocksPerSm));
          if (bps < 1) {
            bps = 1;
          }
          int grid = numSms * bps;
          CUDA_CHECK(cudaMemset(dst, 0, n * sizeof(int64_t)));
          kernel<<<grid, kBlockSize>>>(src, dst, n);
          CUDA_CHECK(cudaDeviceSynchronize());
          int64_t errors = verify(dst, n, reversed, numSms);
          float ms =
              timeMs([&]() { kernel<<<grid, kBlockSize>>>(src, dst, n); });
          double gbps = bytesPerCopy / (ms / 1000.0) / 1.0e9;
          printf(
              "    occ %3.0f%% (%d blk/SM, grid=%d)  %.3f ms  %.1f GB/s  "
              "(%.0f%% nominal)  errors=%ld\n",
              frac * 100.0,
              bps,
              grid,
              ms,
              gbps,
              100.0 * gbps / nominalGbps,
              static_cast<long>(errors));
        }
      };

  printf("\n");
  run("forwardCopy", forwardCopy, 0, /*reversed=*/false);
  run("reverseDirect", reverseDirect, 0, /*reversed=*/true);
  run("reverseShared",
      reverseShared,
      kBlockSize * sizeof(int64_t),
      /*reversed=*/true);

  CUDA_CHECK(cudaFree(src));
  CUDA_CHECK(cudaFree(dst));
  return 0;
}
