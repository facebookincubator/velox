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
//   reverseVector (--vector only) each thread loads one aligned 16-byte vector
//                 (longlong2 = two int64), swaps its two lanes in registers,
//                 and stores it to the mirrored vector slot. Both global
//                 accesses are 128-bit; the reversal never leaves registers, so
//                 no shared memory and half as many threads.
//
// If descending coalesced access is as fast as ascending, reverseDirect matches
// forwardCopy and there is nothing to gain from reverseShared. reverseVector
// tests whether wider (128-bit) transactions sustain bandwidth at lower
// occupancy, where fewer resident warps mean fewer in-flight requests.

#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

DEFINE_bool(
    vector,
    false,
    "Also run reverseVector: a 16-byte (longlong2) reverse that swaps the two "
    "int64 lanes of each aligned vector in registers.");

DEFINE_bool(
    single_block,
    false,
    "Compare scalar (reverseDirect) vs vector (reverseVector) reverse launched "
    "with a single thread block (grid=1), sweeping block dims 128/256/512/1024. "
    "Shows how far one block's in-flight requests get, and whether 128-bit "
    "transactions help when only one block hides latency.");

DEFINE_int32(
    blockdim,
    0,
    "In --single-block mode, test only this block dim instead of sweeping "
    "128/256/512/1024.");

namespace {

constexpr int kBlockSize = 256;
constexpr int kIters = 10;
// A single-block launch over the whole array is slow, so time fewer iterations.
constexpr int kSingleBlockIters = 3;

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

// 16-byte vectorized reverse. Each thread loads one aligned longlong2 (two
// int64), swaps the lanes in registers, and stores it to the mirrored vector
// slot. Source vector v holds src[2v], src[2v+1]; the full reverse sends those
// to dst[n-1-2v], dst[n-2-2v], which are exactly the two lanes of dst vector
// numVec-1-v -- so the destination vector is the source vector with its lanes
// swapped. src/dst are 16-byte aligned (cudaMalloc) and n is even (a multiple
// of kBlockSize), so numVec is exact and no lane spills past the end.
__global__ void reverseVector(const int64_t* src, int64_t* dst, int64_t n) {
  const longlong2* srcVec = reinterpret_cast<const longlong2*>(src);
  longlong2* dstVec = reinterpret_cast<longlong2*>(dst);
  int64_t numVec = n / 2;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < numVec;
       v += stride) {
    longlong2 val = srcVec[v];
    longlong2 rev;
    rev.x = val.y;
    rev.y = val.x;
    dstVec[numVec - 1 - v] = rev;
  }
}

// Same 16-byte reverse, but each thread issues kUnroll independent 128-bit
// loads before any store, raising memory-level parallelism per warp so peak
// bandwidth is reached with fewer resident warps. The k-th load of every thread
// is one grid-stride apart, so each unrolled group stays coalesced; the
// per-lane bound check keeps the tail safe with no past-end access.
template <int kUnroll>
__global__ void
reverseVectorUnroll(const int64_t* src, int64_t* dst, int64_t n) {
  const longlong2* srcVec = reinterpret_cast<const longlong2*>(src);
  longlong2* dstVec = reinterpret_cast<longlong2*>(dst);
  int64_t numVec = n / 2;
  int64_t gridStride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t v = start; v < numVec; v += gridStride * kUnroll) {
    longlong2 vals[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      int64_t vi = v + k * gridStride;
      if (vi < numVec) {
        vals[k] = srcVec[vi];
      }
    }
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      int64_t vi = v + k * gridStride;
      if (vi < numVec) {
        longlong2 rev;
        rev.x = vals[k].y;
        rev.y = vals[k].x;
        dstVec[numVec - 1 - vi] = rev;
      }
    }
  }
}

// Like reverseVector but with restrict pointers (no aliasing, so loads hoist
// ahead of stores) and streaming cache operators (__ldcs/__stcs) that mark the
// accesses evict-first, keeping this once-through data out of L2.
__global__ void reverseVectorStreaming(
    const int64_t* __restrict__ src,
    int64_t* __restrict__ dst,
    int64_t n) {
  const longlong2* srcVec = reinterpret_cast<const longlong2*>(src);
  longlong2* dstVec = reinterpret_cast<longlong2*>(dst);
  int64_t numVec = n / 2;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < numVec;
       v += stride) {
    longlong2 val = __ldcs(&srcVec[v]);
    longlong2 rev;
    rev.x = val.y;
    rev.y = val.x;
    __stcs(&dstVec[numVec - 1 - v], rev);
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

// Times 'iters' launches of 'launch' with CUDA events and returns the average
// milliseconds per launch.
template <typename Launch>
float timeMs(Launch launch, int iters = kIters) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  launch(); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
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

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
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

  // Single-block mode: launch grid=1 and compare scalar (reverseDirect) against
  // vector (reverseVector). One block issues a bounded number of outstanding
  // requests, so this isolates how block size (more warps hiding latency) and
  // 128-bit transactions (more bytes per request) each raise the bandwidth a
  // lone block can reach.
  auto runSingle = [&](const char* name, auto kernel, bool reversed, int bd) {
    CUDA_CHECK(cudaMemset(dst, 0, n * sizeof(int64_t)));
    kernel<<<1, bd>>>(src, dst, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    int64_t errors = verify(dst, n, reversed, numSms);
    float ms =
        timeMs([&]() { kernel<<<1, bd>>>(src, dst, n); }, kSingleBlockIters);
    double gbps = bytesPerCopy / (ms / 1000.0) / 1.0e9;
    printf(
        "    %-14s blockDim=%4d  %.3f ms  %.1f GB/s  (%.1f%% nominal)  "
        "errors=%ld\n",
        name,
        bd,
        ms,
        gbps,
        100.0 * gbps / nominalGbps,
        static_cast<long>(errors));
  };

  printf("\n");
  // cudaMemcpy device-to-device: the tuned contiguous-copy ceiling. It cannot
  // reverse, so it is a forward-copy reference only. Timed with the async form
  // so the launches queue on the stream the events bracket.
  {
    CUDA_CHECK(cudaMemset(dst, 0, n * sizeof(int64_t)));
    CUDA_CHECK(
        cudaMemcpy(dst, src, n * sizeof(int64_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    int64_t errors = verify(dst, n, /*reversed=*/false, numSms);
    float ms = timeMs([&]() {
      cudaMemcpyAsync(
          dst, src, n * sizeof(int64_t), cudaMemcpyDeviceToDevice, 0);
    });
    double gbps = bytesPerCopy / (ms / 1000.0) / 1.0e9;
    printf(
        "  %-14s (forward ceiling)  %.3f ms  %.1f GB/s  (%.0f%% nominal)  "
        "errors=%ld\n",
        "cudaMemcpy",
        ms,
        gbps,
        100.0 * gbps / nominalGbps,
        static_cast<long>(errors));
  }

  if (FLAGS_single_block) {
    auto sweepOne = [&](int bd) {
      printf("  blockDim=%d:\n", bd);
      runSingle("forwardCopy", forwardCopy, /*reversed=*/false, bd);
      runSingle("reverseDirect", reverseDirect, /*reversed=*/true, bd);
      runSingle("reverseVector", reverseVector, /*reversed=*/true, bd);
      runSingle(
          "reverseUnroll2", reverseVectorUnroll<2>, /*reversed=*/true, bd);
      runSingle(
          "reverseUnroll4", reverseVectorUnroll<4>, /*reversed=*/true, bd);
      runSingle("reverseStream", reverseVectorStreaming, /*reversed=*/true, bd);
    };
    printf("Single block (grid=1), scalar vs vector:\n");
    if (FLAGS_blockdim != 0) {
      sweepOne(FLAGS_blockdim);
    } else {
      for (int bd : {128, 256, 512, 1024}) {
        sweepOne(bd);
      }
    }
  } else {
    run("forwardCopy", forwardCopy, 0, /*reversed=*/false);
    run("reverseDirect", reverseDirect, 0, /*reversed=*/true);
    run("reverseShared",
        reverseShared,
        kBlockSize * sizeof(int64_t),
        /*reversed=*/true);
    if (FLAGS_vector) {
      run("reverseVector", reverseVector, 0, /*reversed=*/true);
      run("reverseUnroll2", reverseVectorUnroll<2>, 0, /*reversed=*/true);
      run("reverseUnroll4", reverseVectorUnroll<4>, 0, /*reversed=*/true);
      run("reverseStream", reverseVectorStreaming, 0, /*reversed=*/true);
    }
  }

  CUDA_CHECK(cudaFree(src));
  CUDA_CHECK(cudaFree(dst));
  return 0;
}
