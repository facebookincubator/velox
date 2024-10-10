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

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <nvrtc.h>
#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/Exception.h"

DEFINE_string(
    wavegen_architecture,
    "compute_80",
    "--gpu-architecture flag for generated code");

namespace facebook::velox::wave {

void nvrtcCheck(nvrtcResult result) {
  if (result != NVRTC_SUCCESS) {
    waveError(nvrtcGetErrorString(result));
  }
}

class CompiledModuleImpl : public CompiledModule {
 public:
  CompiledModuleImpl(CUmodule module, std::vector<CUfunction> kernels)
      : module_(module), kernels_(std::move(kernels)) {}

  ~CompiledModuleImpl() {
    auto result = cuModuleUnload(module_);
    if (result != CUDA_SUCCESS) {
      LOG(ERROR) << "Error in unloading module " << result;
    }
  }

  void launch(
      int32_t kernelIdx,
      int32_t numBlocks,
      int32_t numThreads,
      int32_t shared,
      Stream* stream,
      void** args) override;

  KernelInfo info(int32_t kernelIdx) override;

 private:
  CUmodule module_;
  std::vector<CUfunction> kernels_;
};

std::shared_ptr<CompiledModule> CompiledModule::create(const KernelSpec& spec) {
  nvrtcProgram prog;
  nvrtcCreateProgram(
      &prog,
      spec.code.c_str(), // buffer
      spec.filePath.c_str(), // name
      spec.numHeaders, // numHeaders
      spec.headers, // headers
      spec.headerNames); // includeNames
  for (auto& name : spec.entryPoints) {
    nvrtcCheck(nvrtcAddNameExpression(prog, name.c_str()));
  }
  auto architecture =
      fmt::format("--gpu-architecture={}", FLAGS_wavegen_architecture);
  const char* opts[] = {
      architecture.c_str(),
#ifndef NDEBUG
      "-G"
      #else
      "-O3"
#endif
  };
  auto compileResult = nvrtcCompileProgram(
      prog, // prog
      sizeof(opts) / sizeof(char*), // numOptions
      opts); // options

  size_t logSize;

  nvrtcGetProgramLogSize(prog, &logSize);
  std::string log;
  log.resize(logSize);
  nvrtcGetProgramLog(prog, log.data());

  if (compileResult != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    waveError(std::string("Cuda compilation error: ") + log);
  }
  // Obtain PTX from the program.
  size_t ptxSize;
  nvrtcCheck(nvrtcGetPTXSize(prog, &ptxSize));
  std::string ptx;
  ptx.resize(ptxSize);
  nvrtcCheck(nvrtcGetPTX(prog, ptx.data()));
  std::vector<std::string> loweredNames;
  for (auto& entry : spec.entryPoints) {
    const char* temp;
    nvrtcCheck(nvrtcGetLoweredName(prog, entry.c_str(), &temp));
    loweredNames.push_back(std::string(temp));
  }

  nvrtcDestroyProgram(&prog);

  CUmodule module;
  CU_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
  std::vector<CUfunction> funcs;
  for (auto& name : loweredNames) {
    funcs.emplace_back();
    CU_CHECK(cuModuleGetFunction(&funcs.back(), module, name.c_str()));
  }
  return std::make_shared<CompiledModuleImpl>(module, std::move(funcs));
}

void CompiledModuleImpl::launch(
    int32_t kernelIdx,
    int32_t numBlocks,
    int32_t numThreads,
    int32_t shared,
    Stream* stream,
    void** args) {
  auto result = cuLaunchKernel(
      kernels_[kernelIdx],
      numBlocks,
      1,
      1, // grid dim
      numThreads,
      1,
      1, // block dim
      shared,
      reinterpret_cast<CUstream>(stream->stream()->stream),
      args,
      0);
  CU_CHECK(result);
};

KernelInfo CompiledModuleImpl::info(int32_t kernelIdx) {
  KernelInfo info;
  auto f = kernels_[kernelIdx];
  cuFuncGetAttribute(&info.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
  cuFuncGetAttribute(
      &info.sharedMemory, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
  cuFuncGetAttribute(
      &info.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);
  int32_t max;
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&max, f, 256, 0);
  info.maxOccupancy0 = max;
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&max, f, 256, 256 * 32);
  info.maxOccupancy32 = max;
  return info;
}

} // namespace facebook::velox::wave
