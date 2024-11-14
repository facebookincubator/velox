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

#define JITIFY_PRINT_HEADER_PATHS 1

#include <filesystem>
#include "velox/experimental/wave/jit/Headers.h"
#include "velox/external/jitify/jitify.hpp"

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

namespace {

void addFlag(
    const char* flag,
    const char* value,
    int32_t length,
    std::vector<std::string>& data) {
  std::string str(flag);
  str.resize(str.size() + length);
  memcpy(str.data() + strlen(flag), value, length);
  data.push_back(std::move(str));
}

// Gets compiler options from the environment and appends  them  to 'data'.
void getNvrtcOptions(std::vector<std::string>& data) {
  const char* includes = getenv("WAVE_NVRTC_INCLUDE_PATH");
  if (includes && strlen(includes) > 0) {
    for (;;) {
      const char* end = strchr(includes, ':');
      if (!end) {
        addFlag("-I", includes, strlen(includes), data);
        break;
      }
      addFlag("-I", includes, end - includes, data);
      includes = end + 1;
    }
  } else {
    std::string currentPath = std::filesystem::current_path().c_str();
    LOG(INFO) << "Looking for Cuda includes. cwd=" << currentPath
              << " Cuda=" << __CUDA_API_VER_MAJOR__ << "."
              << __CUDA_API_VER_MINOR__;
    auto pathCStr = currentPath.c_str();
    if (auto fbsource = strstr(pathCStr, "fbsource")) {
      // fbcode has cuda includes in fbsource/third-party/cuda/...
      try {
        auto fbsourcePath =
            std::string(pathCStr, fbsource - pathCStr + strlen("fbsource")) +
            "/third-party/cuda";
        LOG(INFO) << "Guessing fbsource path =" << fbsourcePath;
        auto tempPath = fmt::format("/tmp/cuda.{}", getpid());
        auto command = fmt::format(
            "(cd {}; du |grep \"{}\\.{}.*x64-linux.*/cuda$\" |grep -v thrust) >{}",
            fbsourcePath,
            __CUDA_API_VER_MAJOR__,
            __CUDA_API_VER_MINOR__,
            tempPath);
        LOG(INFO) << "Running " << command;
        system(command.c_str());
        std::ifstream result(tempPath);
        std::string line;
        if (!std::getline(result, line)) {
          LOG(ERROR) << "Cuda includes not found in fbcode/third-party";
          return;
        }
        LOG(INFO) << "Got cuda line: " << line;
        // Now trim the size and the trailing /cuda from the line.
        const char* start = strstr(line.c_str(), "./");
        if (!start) {
          LOG(ERROR) << "Line " << line << " does not have ./";
          return;
        }
        auto path = fbsourcePath + "/" + (start + 2);
        // We add the cwd + the found path minus the trailing /cuda.
        addFlag("-I", path.c_str(), path.size() - 5, data);
      } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to infer fbcode Cuda include path: " << e.what();
      }
    } else {
      addFlag(
          "-I",
          "/usr/local/cuda/include",
          strlen("/usr/local/cuda/include"),
          data);
    }
  }
  const char* flags = getenv("WAVE_NVRTC_FLAGS");
  if (flags && strlen(flags)) {
    for (;;) {
      auto end = strchr(flags, ' ');
      if (!end) {
        addFlag("", flags, strlen(flags), data);
        break;
      }
      addFlag("", flags, end - flags, data);
      flags = end + 1;
    }
  }
}

// Contains header names and file contents as a std::string with a trailing 0.
std::vector<std::string> waveHeaderName;
std::vector<std::string> waveHeaderText;

//  data area of waveheader* as a null terminated string.
std::vector<const char*> waveHeaderNameString;
std::vector<const char*> waveHeaderTextString;
std::mutex initMutex;

std::vector<std::string> waveNvrtcFlags;
std::vector<const char*> waveNvrtcFlagsString;

// Adds a trailing zero to make the string.data() a C char*.
void makeNTS(std::string& string) {
  string.resize(string.size() + 1);
  string.back() = 0;
}

void initializeWaveHeaders(
    const std::map<std::string, std::string>& headers,
    const std::string& except) {
  for (auto& pair : headers) {
    if (pair.first == except) {
      continue;
    }
    waveHeaderName.push_back(pair.first);
    makeNTS(waveHeaderName.back());
    waveHeaderText.push_back(pair.second);
    makeNTS(waveHeaderText.back());
  }
  for (auto i = 0; i < waveHeaderName.size(); ++i) {
    waveHeaderNameString.push_back(waveHeaderName[i].data());
    waveHeaderTextString.push_back(waveHeaderText[i].data());
  }
  std::vector<const char*> names;
  std::vector<const char*> text;
  getRegisteredHeaders(names, text);
  for (auto i = 0; i < names.size(); ++i) {
    waveHeaderNameString.push_back(names[i]);
    waveHeaderTextString.push_back(text[i]);
  }
}

// Uses Jitify to compile a sample program on initialization. This
// gathers the JIT-safe includes from Jitify and Cuda and Cub and
// Velox. This also decides flags architecture flags. The Wave kernel
// cache differs from Jitify in having multiple entry points and doing
// background compilation.
void ensureInit() {
  static std::atomic<bool> inited = false;
  if (inited) {
    return;
  }
  std::lock_guard<std::mutex> l(initMutex);

  if (inited) {
    return;
  }

  // Sample kernel that pulls in system headers used by Wave. Checks that key
  // headers are included and uses compile.
  const char* sampleText =
      "Sample\n"
      "#include <cuda/semaphore>\n"
      "__global__ void \n"
      "sampleKernel(unsigned char** bools, int** mtx, int* sizes) { \n"
      "__shared__ int32_t f;\n"
      "typedef cuda::binary_semaphore<cuda::thread_scope_device> Mutex;\n"
      "if (threadIdx.x == 0) {\n"
      "		     f = 1;\n"
      "reinterpret_cast<Mutex*>(&f)->acquire();\n"
      " assert(f == 0);\n"
      " printf(\"pfaal\"); \n"
      "  atomicAdd(&f, 1);\n"
      "}\n"
      "} \n";

  waveNvrtcFlags.push_back("-std=c++17");
#ifndef NDEBUG
  waveNvrtcFlags.push_back("-G");
#else
  // waveNvrtcFlags.push_back("-O3");
#endif
  getNvrtcOptions(waveNvrtcFlags);
  ::jitify::detail::detect_and_add_cuda_arch(waveNvrtcFlags);

  static jitify::JitCache kernel_cache;

  auto program = kernel_cache.program(sampleText, {}, waveNvrtcFlags);

  initializeWaveHeaders(program._impl->_config->sources, "sample");

  for (auto& str : waveNvrtcFlags) {
    makeNTS(str);
    waveNvrtcFlagsString.push_back(str.data());
  }

  inited = true;
}

} // namespace

std::shared_ptr<CompiledModule> CompiledModule::create(const KernelSpec& spec) {
  ensureInit();
  const char** headers = waveHeaderTextString.data();
  const char** headerNames = waveHeaderNameString.data();
  int32_t numHeaders = waveHeaderNameString.size();
  std::vector<const char*> allHeaders;
  std::vector<const char*> allHeaderNames;
  // If spec has extra headers, add them after the system headers.
  if (spec.numHeaders > 0) {
    allHeaderNames = waveHeaderNameString;
    allHeaders = waveHeaderTextString;
    for (auto i = 0; i < spec.numHeaders; ++i) {
      allHeaders.push_back(spec.headers[i]);
      allHeaderNames.push_back(spec.headerNames[i]);
    }
    headers = allHeaders.data();
    headerNames = allHeaderNames.data();
    numHeaders += spec.numHeaders;
  }
  nvrtcProgram prog;
  nvrtcCreateProgram(
      &prog,
      spec.code.c_str(), // buffer
      spec.filePath.c_str(), // name
      numHeaders, // numHeaders
      headers, // headers
      headerNames); // includeNames
  for (auto& name : spec.entryPoints) {
    nvrtcCheck(nvrtcAddNameExpression(prog, name.c_str()));
  }

  auto compileResult = nvrtcCompileProgram(
      prog, // prog
      waveNvrtcFlagsString.size(), // numOptions
      waveNvrtcFlagsString.data()); // options

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
  CUjit_option options[] = {
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  char info[1024];
  char error[1024];
  uint32_t infoSize = sizeof(info);
  uint32_t errorSize = sizeof(error);
  void* values[] = {info, &infoSize, error, &errorSize};

  CUmodule module;
  auto loadResult = cuModuleLoadDataEx(
      &module, ptx.data(), sizeof(values) / sizeof(void*), options, values);
  if (loadResult != CUDA_SUCCESS) {
    LOG(ERROR) << "Load error " << errorSize << " " << infoSize;
    waveError(fmt::format("Error in load module: {} {}", info, error));
  }
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
