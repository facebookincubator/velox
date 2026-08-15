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

#include <cuda.h>
#include <cuda_runtime.h>

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>
#include "velox/experimental/wave/common/Buffer.h"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/Exception.h"
#include "velox/experimental/wave/common/GpuArena.h"
#include "velox/experimental/wave/common/KernelFsCache.h"
#include "velox/experimental/wave/common/tests/BlockTest.h"

#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace facebook::velox::wave {

void testCuCheck(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* str;
    cuGetErrorString(result, &str);
    waveError(std::string("Cuda error: ") + str);
  }
}

class CompileTest : public testing::Test {
 protected:
  void SetUp() override {
    device_ = getDevice();
    setDevice(device_);
    allocator_ = getAllocator(device_);
    arena_ = std::make_unique<GpuArena>(1 << 28, allocator_);
    streams_.push_back(std::make_unique<BlockTestStream>());
  }

  Device* device_;
  GpuAllocator* allocator_;
  std::unique_ptr<GpuArena> arena_;
  std::vector<std::unique_ptr<BlockTestStream>> streams_;
};

struct KernelParams {
  int32_t* array;
  int32_t size;
};

const char* kernelText =
    "using int32_t = int; //#include <cstdint>\n"
    "namespace facebook::velox::wave {\n"
    "  struct KernelParams {\n"
    "    int32_t* array;\n"
    "    int32_t size;\n"
    "  };\n"
    "\n"
    "  void __global__ add1(KernelParams params) {\n"
    "    for (auto i = threadIdx.x; i < params.size; i += blockDim.x) {\n"
    "      ++params.array[i];\n"
    "    }\n"
    "  }\n"
    "\n"
    "  void __global__ add2(KernelParams params) {\n"
    "    for (auto i = threadIdx.x; i < params.size; i += blockDim.x) {\n"
    "      params.array[i] += 2;\n"
    "    }\n"
    "  }\n"
    "} // namespace\n";

void __global__ add3(KernelParams params) {
  for (auto i = threadIdx.x; i < params.size; i += blockDim.x) {
    params.array[i] += 3;
  }
}

TEST_F(CompileTest, module) {
  KernelSpec spec = KernelSpec{
      kernelText,
      {"facebook::velox::wave::add1", "facebook::velox::wave::add2"},
      "/tmp/add1.cu"};
  auto module = CompiledModule::create(spec);
  int32_t* ptr;
  testCuCheck(cuMemAllocManaged(
      reinterpret_cast<CUdeviceptr*>(&ptr),
      1000 * sizeof(int32_t),
      CU_MEM_ATTACH_GLOBAL));
  KernelParams record{ptr, 1000};
  memset(ptr, 0, 1000 * sizeof(int32_t));
  void* recordPtr = &record;
  auto impl = std::make_unique<StreamImpl>();
  testCuCheck(cuStreamCreate((CUstream*)&impl->stream, CU_STREAM_DEFAULT));
  auto stream = std::make_unique<Stream>(std::move(impl));
  module->launch(0, 1, 256, 0, stream.get(), &recordPtr);
  testCuCheck(cuStreamSynchronize((CUstream)stream->stream()->stream));
  EXPECT_EQ(1, ptr[0]);
  auto info = module->info(0);
  EXPECT_EQ(1024, info.maxThreadsPerBlock);

  // See if runtime API kernel works on driver API stream.
  add3<<<1, 256, 0, (cudaStream_t)stream->stream()->stream>>>(record);
  CUDA_CHECK(cudaGetLastError());
  testCuCheck(cuStreamSynchronize((CUstream)stream->stream()->stream));
  EXPECT_EQ(4, ptr[0]);

  auto stream2 = std::make_unique<Stream>();
  module->launch(1, 1, 256, 0, stream2.get(), &recordPtr);
  stream2->wait();
  EXPECT_EQ(6, ptr[0]);
}

TEST_F(CompileTest, cache) {
  KernelSpec spec = KernelSpec{
      kernelText,
      {"facebook::velox::wave::add1", "facebook::velox::wave::add2"},
      "/tmp/add1.cu"};
  auto kernel =
      CompiledKernel::getKernel("add1", [&]() -> KernelSpec { return spec; });
  auto buffer = arena_->allocate<int32_t>(1000);
  memset(buffer->as<int32_t>(), 0, sizeof(int32_t) * 1000);
  KernelParams record{buffer->as<int32_t>(), 1000};
  void* recordPtr = &record;
  auto stream = std::make_unique<Stream>();
  kernel->launch(1, 1, 256, 0, stream.get(), &recordPtr);
  stream->wait();
  EXPECT_EQ(2, buffer->as<int32_t>()[0]);
}

TEST_F(CompileTest, scan) {
  // Tests a warp prefix sum across the warp and then across the 8 first lanes
  // of the warp.

  const char* text =
      "#include \"velox/experimental/wave/common/Scan.cuh\"\n"
      "namespace facebook::velox::wave {\n"
      "__global__ void scanKernel32(int32_t* ints) {\n"
      "  using Scan = WarpScan<uint32_t>;\n"
      "uint32_t out;\n"
      " Scan().exclusiveSum(ints[threadIdx.x], out);\n"
      "ints[threadIdx.x] = out;\n"
      "__syncthreads();\n"
      "}\n"
      "__global__ void scanKernel8(int32_t* ints) {\n"
      "  using Scan = WarpScan<uint32_t, 8>;\n"
      "uint32_t out;\n"
      " Scan().exclusiveSum(ints[threadIdx.x], out);\n"
      "ints[threadIdx.x] = out;\n"
      "__syncthreads();\n"
      "}\n"
      "}\n";

  WaveBufferPtr ints = arena_->allocate<uint32_t>(32);
  for (auto i = 0; i < 32; ++i) {
    ints->as<uint32_t>()[i] = i;
  }
  KernelSpec spec = {
      text,
      {"facebook::velox::wave::scanKernel32",
       "facebook::velox::wave::scanKernel8"},
      "scans.cu"};
  auto module = CompiledModule::create(spec);
  auto stream = std::make_unique<Stream>();
  auto rawInts = ints->as<int32_t>();
  void* params = &rawInts;
  module->launch(0, 1, 32, 0, stream.get(), &params);
  stream->wait();
  int32_t sum = 0;
  for (auto i = 0; i < 32; ++i) {
    EXPECT_EQ(rawInts[i], sum);
    sum += i;
  }

  // test prefix sum over the 8 first lanes.
  for (auto i = 0; i < 32; ++i) {
    rawInts[i] = i;
  }
  module->launch(1, 1, 32, 0, stream.get(), &params);
  stream->wait();
  sum = 0;
  for (auto i = 0; i < 8; ++i) {
    EXPECT_EQ(rawInts[i], i < 8 ? sum : i);
    sum += i;
  }
}

TEST_F(CompileTest, reduce) {
  // Tests a warp reduce.

  const char* text =
      "#include \"velox/experimental/wave/common/Scan.cuh\"\n"
      "namespace facebook::velox::wave {\n"
      "template <typename T> __device__ __forceinline__ T add(T x, T y) {return x + y;}\n"
      "__global__ void reduceKernel32(int32_t* ints, int32_t* result) {\n"
      "  using Reduce = WarpReduce<uint32_t>;\n"
      "  result[threadIdx.x] = Reduce().reduce(ints[threadIdx.x], add<int32_t>);\n"
      "__syncthreads();\n"
      "}\n"
      "}\n";

  WaveBufferPtr ints = arena_->allocate<uint32_t>(32);
  for (auto i = 0; i < 32; ++i) {
    ints->as<uint32_t>()[i] = i;
  }
  WaveBufferPtr result = arena_->allocate<uint32_t>(32);

  KernelSpec spec = {
      text, {"facebook::velox::wave::reduceKernel32"}, "reduces.cu"};
  auto module = CompiledModule::create(spec);
  auto ptr1 = ints->as<int32_t>();
  auto ptr2 = result->as<int32_t>();
  auto stream = std::make_unique<Stream>();
  int32_t** arrays[2] = {&ptr1, &ptr2};
  module->launch(0, 1, 32, 0, stream.get(), reinterpret_cast<void**>(arrays));
  stream->wait();
  int32_t sum = 0;
  for (auto i = 0; i < 32; ++i) {
    sum += i;
  }
  EXPECT_EQ(sum, ptr2[0]);
}

TEST_F(CompileTest, cubinCache) {
  const char* text =
      "#include \"velox/experimental/wave/common/Scan.cuh\"\n"
      "namespace facebook::velox::wave {\n"
      "__global__ void scanKernel32(int32_t* ints) {\n"
      "  using Scan = WarpScan<uint32_t>;\n"
      "uint32_t out;\n"
      " Scan().exclusiveSum(ints[threadIdx.x], out);\n"
      "ints[threadIdx.x] = out;\n"
      "__syncthreads();\n"
      "}\n"
      "}\n";

  std::string cubinPath =
      "/tmp/compile_test_scan_" + std::to_string(getpid()) + ".cubin";

  // Phase 1: compile via getKernel with cubinPath and postCompile set.
  bool postCompileCalled = false;
  std::vector<std::string> savedLoweredNames;

  auto kernel = CompiledKernel::getKernel("scanCubinTest", [&]() -> KernelSpec {
    KernelSpec spec;
    spec.code = text;
    spec.entryPoints = {"facebook::velox::wave::scanKernel32"};
    spec.filePath = "scan_cubin.cu";
    spec.cubinPath = cubinPath;
    spec.postCompile = [&](KernelSpec& compiled, std::exception_ptr error) {
      EXPECT_EQ(error, nullptr);
      postCompileCalled = true;
      savedLoweredNames = compiled.loweredNames;
    };
    return spec;
  });

  // Verify postCompile was called and loweredNames were populated.
  auto info = kernel->info(0);
  EXPECT_GT(info.numRegs, 0);
  EXPECT_TRUE(postCompileCalled);
  ASSERT_EQ(savedLoweredNames.size(), 1);

  // Launch and verify result.
  auto runAndCheck = [&](CompiledKernel& kernel) {
    WaveBufferPtr ints = arena_->allocate<uint32_t>(32);
    for (auto i = 0; i < 32; ++i) {
      ints->as<uint32_t>()[i] = i;
    }
    auto rawInts = ints->as<int32_t>();
    void* params = &rawInts;
    auto stream = std::make_unique<Stream>();
    kernel.launch(0, 1, 32, 0, stream.get(), &params);
    stream->wait();
    int32_t sum = 0;
    for (auto i = 0; i < 32; ++i) {
      EXPECT_EQ(rawInts[i], sum);
      sum += i;
    }
  };
  runAndCheck(*kernel);

  // Phase 2: clear cache and load from CUBIN via getKernel with
  // fromCubinPath.
  CompiledKernel::clearCache();
  auto kernel2 =
      CompiledKernel::getKernel("scanCubinTest2", [&]() -> KernelSpec {
        KernelSpec spec;
        spec.entryPoints = {"facebook::velox::wave::scanKernel32"};
        spec.fromCubinPath = cubinPath;
        spec.loweredNames = savedLoweredNames;
        return spec;
      });
  runAndCheck(*kernel2);
}

TEST_F(CompileTest, fsCache) {
  const char* scanText =
      "#include \"velox/experimental/wave/common/Scan.cuh\"\n"
      "namespace facebook::velox::wave {\n"
      "__global__ void scanKernel32(int32_t* ints) {\n"
      "  using Scan = WarpScan<uint32_t>;\n"
      "uint32_t out;\n"
      " Scan().exclusiveSum(ints[threadIdx.x], out);\n"
      "ints[threadIdx.x] = out;\n"
      "__syncthreads();\n"
      "}\n"
      "}\n";

  auto cachePath = "/tmp/compile_test_fscache_" + std::to_string(getpid());
  std::error_code ec;
  std::filesystem::remove_all(cachePath, ec);
  SCOPE_EXIT {
    std::filesystem::remove_all(cachePath, ec);
  };

  auto runAndCheck = [&](CompiledKernel& kernel) {
    WaveBufferPtr ints = arena_->allocate<uint32_t>(32);
    for (auto i = 0; i < 32; ++i) {
      ints->as<uint32_t>()[i] = i;
    }
    auto rawInts = ints->as<int32_t>();
    void* params = &rawInts;
    auto stream = std::make_unique<Stream>();
    kernel.launch(0, 1, 32, 0, stream.get(), &params);
    stream->wait();
    int32_t sum = 0;
    for (auto i = 0; i < 32; ++i) {
      EXPECT_EQ(rawInts[i], sum);
      sum += i;
    }
  };

  auto makeGenFunc = [&]() -> KernelGenFunc {
    return [&]() -> KernelSpec {
      KernelSpec spec;
      spec.code = scanText;
      spec.entryPoints = {"facebook::velox::wave::scanKernel32"};
      spec.filePath = "scan_fs.cu";
      return spec;
    };
  };

  {
    KernelFsCache cache(cachePath);

    // First call: expect miss.
    {
      auto kernel = cache.getKernel(scanText, makeGenFunc());
      runAndCheck(*kernel);
    }
    EXPECT_EQ(cache.misses(), 1);
    EXPECT_EQ(cache.hits(), 0);
    EXPECT_EQ(cache.size(), 1);

    // Second call after clearing RAM cache: expect fs cache hit.
    CompiledKernel::clearCache();
    {
      auto kernel2 = cache.getKernel(scanText, makeGenFunc());
      runAndCheck(*kernel2);
    }
    EXPECT_EQ(cache.hits(), 1);
    EXPECT_EQ(cache.misses(), 1);
    EXPECT_EQ(cache.size(), 1);
  }

  // Destroy the fs cache and RAM cache, then recreate from the same path.
  CompiledKernel::clearCache();
  {
    KernelFsCache cache2(cachePath);
    auto kernel3 = cache2.getKernel(scanText, makeGenFunc());
    runAndCheck(*kernel3);
    EXPECT_EQ(cache2.hits(), 1);
    EXPECT_EQ(cache2.misses(), 0);
    EXPECT_EQ(cache2.size(), 1);

    // Make a broken kernel text that will fail to compile.
    auto badText = std::string(scanText) + "#error intentional\n";
    auto sizeBefore = cache2.size();
    bool threw = false;
    try {
      auto badKernel = cache2.getKernel(badText, [&]() -> KernelSpec {
        KernelSpec spec;
        spec.code = badText;
        spec.entryPoints = {"facebook::velox::wave::scanKernel32"};
        spec.filePath = "scan_bad.cu";
        return spec;
      });
      // Access info to force the deferred compilation to complete.
      badKernel->info(0);
    } catch (...) {
      threw = true;
    }
    EXPECT_TRUE(threw);
    EXPECT_EQ(cache2.size(), sizeBefore);
  }
}

} // namespace facebook::velox::wave
