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
#include <gtest/gtest.h>
#include <nvrtc.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace torch::wave {
namespace {

// Compile CUDA source with NVRTC, save the CUBIN to a file, then load it back
// with cuModuleLoad and return a CUfunction.
CUfunction compileAndLoad(
    const char* source,
    const char* funcName,
    const std::string& cubinPath) {
  // NVRTC has no filesystem access, so provide standard integer typedefs.
  static const char* kStdintHeader =
      "typedef signed char int8_t;\n"
      "typedef short int16_t;\n"
      "typedef int int32_t;\n"
      "typedef long long int64_t;\n"
      "typedef unsigned char uint8_t;\n"
      "typedef unsigned short uint16_t;\n"
      "typedef unsigned int uint32_t;\n"
      "typedef unsigned long long uint64_t;\n";
  static const char* kStdintName = "stdint.h";
  nvrtcProgram prog;
  nvrtcCreateProgram(
      &prog, source, "kernel.cu", 1, &kStdintHeader, &kStdintName);
  nvrtcAddNameExpression(prog, funcName);

  CUdevice device;
  cuDeviceGet(&device, 0);
  int major, minor;
  cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  auto arch =
      "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
  const char* opts[] = {arch.c_str(), "-std=c++17"};
  auto compileResult = nvrtcCompileProgram(prog, 2, opts);
  if (compileResult != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    nvrtcGetProgramLog(prog, log.data());
    nvrtcDestroyProgram(&prog);
    EXPECT_TRUE(false) << "NVRTC compile failed: " << log;
    return nullptr;
  }

  // Get the lowered (mangled) name.
  const char* loweredName;
  nvrtcGetLoweredName(prog, funcName, &loweredName);
  std::string mangledName(loweredName);

  // Extract CUBIN.
  size_t cubinSize;
  nvrtcGetCUBINSize(prog, &cubinSize);
  std::vector<char> cubin(cubinSize);
  nvrtcGetCUBIN(prog, cubin.data());
  nvrtcDestroyProgram(&prog);

  // Write CUBIN to file.
  {
    std::ofstream out(cubinPath, std::ios::binary);
    out.write(cubin.data(), cubin.size());
  }
  printf("Wrote %zu byte CUBIN to %s\n", cubinSize, cubinPath.c_str());

  // Load from file with cuModuleLoad.
  CUmodule module;
  auto loadResult = cuModuleLoad(&module, cubinPath.c_str());
  EXPECT_EQ(loadResult, CUDA_SUCCESS) << "cuModuleLoad failed: " << loadResult;

  CUfunction func;
  auto getResult = cuModuleGetFunction(&func, module, mangledName.c_str());
  EXPECT_EQ(getResult, CUDA_SUCCESS)
      << "cuModuleGetFunction failed: " << getResult;
  return func;
}

class KernelLdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (cuInit(0) != CUDA_SUCCESS) {
      GTEST_SKIP() << "cuInit failed, no CUDA available";
    }
    CUdevice device;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
      GTEST_SKIP() << "No CUDA device available";
    }
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    if (!ctx) {
      if (cuCtxCreate(&ctx, 0, device) != CUDA_SUCCESS) {
        GTEST_SKIP() << "Failed to create CUDA context";
      }
    }
  }
};

TEST_F(KernelLdTest, vectorAdd) {
  constexpr int32_t kSize = 10000;

  const char* source = R"(
extern "C" __global__ void vectorAdd(
    const float* a, const float* b, float* c, int n) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
)";

  auto func =
      compileAndLoad(source, "vectorAdd", "/tmp/kernel_ld_test_add.cubin");
  ASSERT_NE(func, nullptr);

  CUdeviceptr dA, dB, dC;
  cuMemAlloc(&dA, kSize * sizeof(float));
  cuMemAlloc(&dB, kSize * sizeof(float));
  cuMemAlloc(&dC, kSize * sizeof(float));

  std::vector<float> hA(kSize), hB(kSize);
  for (int i = 0; i < kSize; ++i) {
    hA[i] = static_cast<float>(i);
    hB[i] = static_cast<float>(i * 2);
  }
  cuMemcpyHtoD(dA, hA.data(), kSize * sizeof(float));
  cuMemcpyHtoD(dB, hB.data(), kSize * sizeof(float));

  int n = kSize;
  void* args[] = {&dA, &dB, &dC, &n};
  int blockSize = 256;
  int gridSize = (kSize + blockSize - 1) / blockSize;
  auto launchResult = cuLaunchKernel(
      func, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
  ASSERT_EQ(launchResult, CUDA_SUCCESS) << "cuLaunchKernel failed";
  cuCtxSynchronize();

  std::vector<float> hC(kSize);
  cuMemcpyDtoH(hC.data(), dC, kSize * sizeof(float));

  for (int i = 0; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(hC[i], hA[i] + hB[i]) << "Mismatch at " << i;
  }

  cuMemFree(dA);
  cuMemFree(dB);
  cuMemFree(dC);
}

TEST_F(KernelLdTest, blockReduce) {
  constexpr int32_t kSize = 1024;
  constexpr int32_t kBlockSize = 256;

  const char* source = R"(
extern "C" __global__ void blockReduce(
    const float* input, float* output, int n) {
  __shared__ float shared[256];
  auto tid = threadIdx.x;
  auto idx = blockIdx.x * blockDim.x + tid;
  shared[tid] = (idx < static_cast<unsigned int>(n)) ? input[idx] : 0.0f;
  __syncthreads();
  for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}
)";

  auto func =
      compileAndLoad(source, "blockReduce", "/tmp/kernel_ld_test_reduce.cubin");
  ASSERT_NE(func, nullptr);

  int gridSize = (kSize + kBlockSize - 1) / kBlockSize;
  CUdeviceptr dInput, dOutput;
  cuMemAlloc(&dInput, kSize * sizeof(float));
  cuMemAlloc(&dOutput, gridSize * sizeof(float));

  std::vector<float> hInput(kSize, 1.0f);
  cuMemcpyHtoD(dInput, hInput.data(), kSize * sizeof(float));

  int n = kSize;
  void* args[] = {&dInput, &dOutput, &n};
  auto launchResult = cuLaunchKernel(
      func, gridSize, 1, 1, kBlockSize, 1, 1, 0, nullptr, args, nullptr);
  ASSERT_EQ(launchResult, CUDA_SUCCESS);
  cuCtxSynchronize();

  std::vector<float> hOutput(gridSize);
  cuMemcpyDtoH(hOutput.data(), dOutput, gridSize * sizeof(float));

  for (int i = 0; i < gridSize; ++i) {
    int blockElements = std::min(kBlockSize, kSize - i * kBlockSize);
    EXPECT_FLOAT_EQ(hOutput[i], static_cast<float>(blockElements))
        << "Block " << i;
  }

  cuMemFree(dInput);
  cuMemFree(dOutput);
}

// Verify that a CUBIN saved from a previous test can be loaded again without
// recompilation.
TEST_F(KernelLdTest, loadCachedCubin) {
  auto cubinPath = "/tmp/kernel_ld_test_add.cubin";
  std::ifstream check(cubinPath, std::ios::binary);
  if (!check.good()) {
    GTEST_SKIP() << "No cached CUBIN from vectorAdd test";
  }
  check.close();

  CUmodule module;
  auto loadResult = cuModuleLoad(&module, cubinPath);
  ASSERT_EQ(loadResult, CUDA_SUCCESS) << "cuModuleLoad cached failed";

  CUfunction func;
  auto getResult = cuModuleGetFunction(&func, module, "vectorAdd");
  ASSERT_EQ(getResult, CUDA_SUCCESS) << "cuModuleGetFunction cached failed";

  // Quick sanity: launch with 1 element.
  CUdeviceptr dA, dB, dC;
  cuMemAlloc(&dA, sizeof(float));
  cuMemAlloc(&dB, sizeof(float));
  cuMemAlloc(&dC, sizeof(float));
  float one = 1.0f, two = 2.0f;
  cuMemcpyHtoD(dA, &one, sizeof(float));
  cuMemcpyHtoD(dB, &two, sizeof(float));

  int n = 1;
  void* args[] = {&dA, &dB, &dC, &n};
  cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
  cuCtxSynchronize();

  float result;
  cuMemcpyDtoH(&result, dC, sizeof(float));
  EXPECT_FLOAT_EQ(result, 3.0f);

  cuMemFree(dA);
  cuMemFree(dB);
  cuMemFree(dC);
}

} // namespace
} // namespace torch::wave
