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

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h> // @manual
#include <folly/init/Init.h>
#include <glog/logging.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/SerialGraphExecutor.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/kernels/KernelFactory.h>

#include <caffe2/caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Pt2Load.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/WaveConfig.h"

DEFINE_int32(num_repeats, 1, "Number of timed repetitions for each run type");
DEFINE_bool(
    standalone_kernels,
    false,
    "Treat all operations as standalone kernels");
DEFINE_int32(block_dim, 256, "CUDA thread block size");
DEFINE_int32(
    single_block,
    -1,
    "Force single block grid: -1=auto, 0=multi, 1=single");
DEFINE_string(
    custom,
    "",
    "Custom test model base name (without .pt2/.pt extension)");
DEFINE_bool(wave_only, false, "Skip serial CPU and serial GPU execution");
DEFINE_bool(
    standalone_timing,
    false,
    "Print standalone execution times after wave run");

namespace torch::wave {
namespace {

/// Returns the path to a test data file. In fbcode (Buck), the CWD ends with
/// "fbcode" and files are found via baseDir/filePath. In CMake builds the CWD
/// is set to the source directory containing the test, so just filePath works.
std::string getDataFilePath(
    const std::string& baseDir,
    const std::string& filePath) {
  auto cwd = std::filesystem::current_path().string();
  if (cwd.size() >= 6 && cwd.compare(cwd.size() - 6, 6, "fbcode") == 0) {
    return cwd + "/" + baseDir + "/" + filePath;
  }
  return cwd + "/" + filePath;
}

using Clock = std::chrono::high_resolution_clock;

/// Timing results from running a model through an executor.
struct RunTiming {
  /// Time to transfer inputs to device (0 for CPU executors).
  int64_t dataTransferUs{0};
  /// Execution time excluding data transfer.
  int64_t executeUs{0};
  /// Per-launch debug info (outer = launches, inner = blocks).
  std::vector<std::vector<DebugInfo>> debugInfo;
};

/// Reads a file into a vector of chars.
std::vector<char> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.good()) << "Cannot open " << path;
  auto size = file.tellg();
  file.seekg(0);
  std::vector<char> data(size);
  file.read(data.data(), size);
  return data;
}

/// Loads reference tensors saved by Python via torch.save([t1, t2, ...], path).
std::vector<at::Tensor> loadReferenceTensors(const std::string& path) {
  auto data = readFile(path);
  auto ivalue = torch::jit::pickle_load(data);
  EXPECT_TRUE(ivalue.isList()) << "Expected a list of tensors";
  std::vector<at::Tensor> tensors;
  for (const auto& item : ivalue.toListRef()) {
    EXPECT_TRUE(item.isTensor());
    tensors.push_back(item.toTensor());
  }
  return tensors;
}

/// Extracts tensors from IValues and transfers them to device.
/// Returns the device tensors and the transfer time in microseconds.
std::pair<std::vector<at::Tensor>, int64_t> inputsToDevice(
    std::vector<c10::IValue>& inputs) {
  std::vector<at::Tensor> hostTensors;
  for (auto& iv : inputs) {
    if (iv.isTensor()) {
      hostTensors.push_back(iv.toTensor());
    }
  }

  facebook::velox::wave::Stream stream;
  std::vector<at::Tensor> deviceTensors;

  auto start = Clock::now();
  tensorsToDevice(hostTensors, deviceTensors, stream);
  stream.wait();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start)
                .count();

  return {std::move(deviceTensors), us};
}

/// Extracts tensors from output IValues and transfers them to host.
std::vector<at::Tensor> outputsToHost(
    const std::vector<c10::IValue>& outputs,
    const std::string& label) {
  std::vector<at::Tensor> deviceTensors;
  for (const auto& iv : outputs) {
    EXPECT_TRUE(iv.isTensor()) << label << " output is not a tensor";
    if (iv.isTensor()) {
      deviceTensors.push_back(iv.toTensor());
    }
  }

  facebook::velox::wave::Stream stream;
  std::vector<at::Tensor> hostTensors;
  tensorsToHost(deviceTensors, hostTensors, stream);
  stream.wait();

  return hostTensors;
}

/// Holds the loaded model and weights shared across test methods.
struct ModelFixture {
  std::string pt2Path;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader;
  LoadedModel model;
  std::shared_ptr<nativert::Weights> weights;
  nativert::ExecutorConfig config;

  static std::unique_ptr<ModelFixture> load(const std::string& pt2Path) {
    auto fixture = std::make_unique<ModelFixture>();
    fixture->pt2Path = pt2Path;

    fixture->reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
        std::make_unique<caffe2::serialize::FileAdapter>(pt2Path));

    auto modelNames = getModelNames(*fixture->reader);
    EXPECT_FALSE(modelNames.empty()) << "No models found in " << pt2Path;
    if (modelNames.empty()) {
      return nullptr;
    }

    fixture->model = loadPt2Model(fixture->reader, modelNames[0]);
    const auto& graph = *fixture->model.graph;

    fixture->weights = std::make_shared<nativert::Weights>(
        &graph,
        fixture->reader,
        fixture->model.tensorPaths,
        torch::_export::archive_spec::WEIGHTS_DIR,
        fixture->model.constantPaths,
        torch::_export::archive_spec::CONSTANTS_DIR);

    return fixture;
  }

  /// Creates fresh node kernels (each executor consumes them).
  std::vector<std::unique_ptr<nativert::OpKernel>> makeKernels() const {
    nativert::KernelFactory factory;
    auto execKernels =
        factory.initializeNodeKernels(*model.graph, weights, config, reader);
    return std::move(execKernels.nodeKernels);
  }
};

/// Returns a debug string showing up to the first 10 elements of a tensor
/// after flattening to 1-D, plus the shape.
std::string tensorDebugString(const at::Tensor& t) {
  auto flat = t.cpu().contiguous().flatten();
  auto n = std::min<int64_t>(flat.numel(), 10);
  std::stringstream ss;
  ss << "shape=" << t.sizes() << " [";
  for (int64_t i = 0; i < n; ++i) {
    if (i > 0) {
      ss << ", ";
    }
    if (flat.scalar_type() == c10::ScalarType::Float) {
      ss << flat[i].item<float>();
    } else if (flat.scalar_type() == c10::ScalarType::Double) {
      ss << flat[i].item<double>();
    } else if (flat.scalar_type() == c10::ScalarType::Long) {
      ss << flat[i].item<int64_t>();
    } else if (flat.scalar_type() == c10::ScalarType::Int) {
      ss << flat[i].item<int32_t>();
    } else if (flat.scalar_type() == c10::ScalarType::Half) {
      ss << flat[i].item<at::Half>();
    } else if (flat.scalar_type() == c10::ScalarType::BFloat16) {
      ss << flat[i].item<at::BFloat16>();
    } else {
      ss << flat[i].item<float>();
    }
  }
  if (flat.numel() > n) {
    ss << ", ...";
  }
  ss << "]";
  return ss.str();
}

/// Prints standalone execution times from a WaveGraphExecutor, sorted highest
/// time first. Each node is serialized as in Launch::toString.
void standaloneReport(WaveGraphExecutor& executor) {
  auto stats = executor.getStandaloneStats();
  if (stats.empty()) {
    return;
  }
  std::sort(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });
  LOG(INFO) << "Standalone timing:";
  for (const auto& [name, micros] : stats) {
    LOG(INFO) << "  " << micros << " us  " << name;
  }
}

class ExecutorTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Diagnostic: print device properties and test basic CUDA ops.
    LOG(INFO) << "CUDA device count: " << at::cuda::device_count();
    auto* props = at::cuda::getDeviceProperties(0);
    LOG(INFO) << "Device 0: " << props->name << " sm_" << props->major
              << props->minor;

    // Test empty (no kernel launch) vs zeros (requires fill kernel).
    LOG(INFO) << "Testing at::empty on CUDA...";
    try {
      auto e = at::empty(
          {4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
      LOG(INFO) << "at::empty OK, data_ptr=" << e.data_ptr();
    } catch (const std::exception& e) {
      LOG(ERROR) << "at::empty FAILED: " << e.what();
    }

    LOG(INFO) << "Testing at::ones on CUDA (fill kernel)...";
    try {
      auto o = at::ones(
          {4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
      LOG(INFO) << "at::ones OK: " << o.cpu().sum().item<float>();
    } catch (const std::exception& e) {
      LOG(ERROR) << "at::ones FAILED: " << e.what();
    }

    WaveConfig::get().allStandalone = FLAGS_standalone_kernels;
    WaveConfig::get().blockSize = FLAGS_block_dim;
    if (FLAGS_single_block >= 0) {
      WaveConfig::get().useSingleBlock = FLAGS_single_block != 0;
    }

    initialize();
  }

  /// Runs a .pt2 model through the nativert SerialGraphExecutor on CPU.
  /// Verifies outputs match 'expected' and returns timing.
  RunTiming runSerial(
      ModelFixture& fixture,
      const std::vector<at::Tensor>& expected) {
    const auto& graph = *fixture.model.graph;

    auto kernels = fixture.makeKernels();
    nativert::SerialGraphExecutor executor(
        graph, std::move(kernels), fixture.config);

    auto frame = std::make_unique<nativert::ExecutionFrame>(
        graph, *fixture.weights, fixture.config);

    auto inputs = loadSampleInputs(fixture);

    auto start = Clock::now();
    auto outputs = executor.execute(*frame, std::move(inputs));
    auto executeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - start)
                         .count();

    verifyOutputs(outputs, expected, "serial-pass1");

    // Run a second time with the same frame to test frame reuse.
    auto inputs2 = loadSampleInputs(fixture);
    auto outputs2 = executor.execute(*frame, std::move(inputs2));
    verifyOutputs(outputs2, expected, "serial-pass2");

    return {0, executeUs, {}};
  }

  RunTiming runSerialOnDevice(
      ModelFixture& fixture,
      const std::vector<at::Tensor>& expected) {
    const auto& graph = *fixture.model.graph;

    auto kernels = fixture.makeKernels();
    nativert::SerialGraphExecutor executor(
        graph, std::move(kernels), fixture.config);

    auto frame = std::make_unique<nativert::ExecutionFrame>(
        graph, *fixture.weights, fixture.config);

    auto inputs = loadSampleInputs(fixture);
    auto [deviceTensors, dataMovUs] = inputsToDevice(inputs);

    // Build IValue inputs from device tensors.
    std::vector<c10::IValue> deviceIvalues;
    deviceIvalues.reserve(deviceTensors.size());
    for (auto& t : deviceTensors) {
      deviceIvalues.emplace_back(std::move(t));
    }

    auto start = Clock::now();
    auto outputs = executor.execute(*frame, std::move(deviceIvalues));
    auto executeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - start)
                         .count();

    auto hostOutputs = outputsToHost(outputs, "serial-gpu");
    verifyOutputs(hostOutputs, expected, "serial-gpu");
    return {dataMovUs, executeUs, {}};
  }

  /// Fills device inputs into a WaveGraphExecutor frame.
  void fillWaveFrame(
      const nativert::Graph& graph,
      nativert::ExecutionFrame& frame,
      const std::vector<at::Tensor>& deviceInputs) {
    const auto& userInputNames = graph.signature().userInputs();
    size_t tensorIdx = 0;
    for (const auto& name : userInputNames) {
      auto* value = graph.tryGetValue(name);
      EXPECT_NE(value, nullptr) << "No value for input " << name;
      if (value && tensorIdx < deviceInputs.size()) {
        frame.setIValue(
            value->id(), c10::IValue(deviceInputs[tensorIdx++].clone()));
      }
    }
  }

  /// Runs a .pt2 model through the WaveGraphExecutor on device.
  /// Verifies outputs match 'expected' and returns timing.
  RunTiming runWave(
      ModelFixture& fixture,
      const std::vector<at::Tensor>& expected) {
    auto& graph = *fixture.model.graph;

    WaveGraphExecutor waveExec(
        graph, fixture.makeKernels(), fixture.config, fixture.weights);

    auto pooledFrame = waveExec.getFrame();
    EXPECT_NE(pooledFrame, nullptr);

    auto inputs = loadSampleInputs(fixture);
    auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

    fillWaveFrame(graph, *pooledFrame, deviceInputs);

    auto waveStart = Clock::now();
    auto waveOutputs = waveExec.executeWithPrefilledFrame(*pooledFrame);
    auto waveUs = std::chrono::duration_cast<std::chrono::microseconds>(
                      Clock::now() - waveStart)
                      .count();

    auto debugInfo = waveExec.getDebugInfo();

    waveExec.returnFrame(std::move(pooledFrame));

    auto hostOutputs = outputsToHost(waveOutputs, "wave");
    verifyOutputs(hostOutputs, expected, "wave");
    return {dataMovUs, waveUs, std::move(debugInfo)};
  }

  /// Loads a .pt2 model and reference results, runs serial on CPU, serial on
  /// device and wave, and logs the run times for each.
  void runTest(
      const std::string& pt2File,
      const std::string& resultsFile,
      const std::string& label = "") {
    const std::string kBaseDir = "velox/experimental/torchwave/tests";
    auto pt2Path =
        pt2File[0] == '/' ? pt2File : getDataFilePath(kBaseDir, pt2File);
    auto resultsPath = resultsFile[0] == '/'
        ? resultsFile
        : getDataFilePath(kBaseDir, resultsFile);
    auto displayName = label.empty() ? pt2File : pt2File + " [" + label + "]";
    auto fixture = ModelFixture::load(pt2Path);
    ASSERT_NE(fixture, nullptr);

    auto expected = loadReferenceTensors(resultsPath);
    ASSERT_FALSE(expected.empty());

    const int repeats = FLAGS_num_repeats;

    if (!FLAGS_wave_only) {
      // Serial CPU.
      auto serialCold = runSerial(*fixture, expected);
      int64_t serialMin = INT64_MAX, serialMax = 0, serialSum = 0;
      std::vector<int64_t> serialTimes(repeats);
      for (int i = 0; i < repeats; ++i) {
        auto t = runSerial(*fixture, expected);
        serialTimes[i] = t.executeUs;
        serialMin = std::min(serialMin, t.executeUs);
        serialMax = std::max(serialMax, t.executeUs);
        serialSum += t.executeUs;
      }
      std::sort(serialTimes.begin(), serialTimes.end());
      auto serialP90 = serialTimes[repeats * 90 / 100];
      LOG(INFO) << displayName << " serial CPU (" << repeats
                << " repeats): min=" << serialMin
                << " avg=" << serialSum / repeats << " p90=" << serialP90
                << " max=" << serialMax << " us (cold=" << serialCold.executeUs
                << " us)";

      // Serial GPU.
      auto deviceCold = runSerialOnDevice(*fixture, expected);
      int64_t deviceMin = INT64_MAX, deviceMax = 0, deviceSum = 0;
      std::vector<int64_t> deviceTimes(repeats);
      for (int i = 0; i < repeats; ++i) {
        auto t = runSerialOnDevice(*fixture, expected);
        deviceTimes[i] = t.executeUs;
        deviceMin = std::min(deviceMin, t.executeUs);
        deviceMax = std::max(deviceMax, t.executeUs);
        deviceSum += t.executeUs;
      }
      std::sort(deviceTimes.begin(), deviceTimes.end());
      auto deviceP90 = deviceTimes[repeats * 90 / 100];
      LOG(INFO) << displayName
                << " serial GPU H2D: cold=" << deviceCold.dataTransferUs
                << " us";
      LOG(INFO) << displayName << " serial GPU (" << repeats
                << " repeats): min=" << deviceMin
                << " avg=" << deviceSum / repeats << " p90=" << deviceP90
                << " max=" << deviceMax << " us (cold=" << deviceCold.executeUs
                << " us)";
    }

    // Wave.
    auto& graph = *fixture->model.graph;
    WaveGraphExecutor waveExec(
        graph, fixture->makeKernels(), fixture->config, fixture->weights);

    auto pooledFrame = waveExec.getFrame();
    ASSERT_NE(pooledFrame, nullptr);

    auto inputs = loadSampleInputs(*fixture);
    auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

    // Initial run: fill frame, execute, verify.
    fillWaveFrame(graph, *pooledFrame, deviceInputs);

    auto waveStart = Clock::now();
    auto waveOutputs = waveExec.executeWithPrefilledFrame(*pooledFrame);
    auto waveColdUs = std::chrono::duration_cast<std::chrono::microseconds>(
                          Clock::now() - waveStart)
                          .count();

    auto hostOutputs = outputsToHost(waveOutputs, "wave");
    verifyOutputs(hostOutputs, expected, "wave");

    // Repeated runs: refill inputs and re-execute only.
    int64_t waveMin = INT64_MAX, waveMax = 0, waveSum = 0;
    std::vector<int64_t> waveTimes(repeats);
    std::vector<std::vector<DebugInfo>> lastDebugInfo;
    for (int i = 0; i < repeats; ++i) {
      fillWaveFrame(graph, *pooledFrame, deviceInputs);

      auto start = Clock::now();
      waveExec.executeWithPrefilledFrame(*pooledFrame);
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - start)
                    .count();
      waveTimes[i] = us;
      waveMin = std::min(waveMin, us);
      waveMax = std::max(waveMax, us);
      waveSum += us;
      if (i == repeats - 1) {
        lastDebugInfo = waveExec.getDebugInfo();
      }
    }

    waveExec.returnFrame(std::move(pooledFrame));

    std::sort(waveTimes.begin(), waveTimes.end());
    auto waveP90 = waveTimes[repeats * 90 / 100];
    LOG(INFO) << displayName << " wave H2D: cold=" << dataMovUs << " us";
    LOG(INFO) << displayName << " wave (" << repeats
              << " repeats): min=" << waveMin << " avg=" << waveSum / repeats
              << " p90=" << waveP90 << " max=" << waveMax
              << " us (cold=" << waveColdUs << " us)";

    for (size_t li = 0; li < lastDebugInfo.size(); ++li) {
      const auto& blocks = lastDebugInfo[li];
      struct OpStats {
        int64_t count{0};
        int64_t minClocks{std::numeric_limits<int64_t>::max()};
        int64_t maxClocks{0};
        int64_t sumClocks{0};
      };
      std::map<int32_t, OpStats> opMap;
      for (const auto& b : blocks) {
        auto& s = opMap[b.op];
        s.count++;
        s.minClocks = std::min(s.minClocks, b.clocks);
        s.maxClocks = std::max(s.maxClocks, b.clocks);
        s.sumClocks += b.clocks;
      }
      std::vector<std::pair<int32_t, OpStats>> sorted(
          opMap.begin(), opMap.end());
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return (a.second.sumClocks / a.second.count) <
            (b.second.sumClocks / b.second.count);
      });
      std::stringstream ss;
      ss << displayName << " launch " << li << ": ";
      bool first = true;
      for (const auto& [op, s] : sorted) {
        if (!first) {
          ss << ", ";
        }
        first = false;
        ss << "op " << op << " count: " << s.count << " " << s.minClocks << "/"
           << (s.sumClocks / s.count) << "/" << s.maxClocks;
      }
      LOG(INFO) << ss.str();
    }

    if (FLAGS_standalone_timing) {
      standaloneReport(waveExec);
    }
  }

 private:
  /// Loads sample inputs from the .pt2 package.
  std::vector<c10::IValue> loadSampleInputs(ModelFixture& fixture) {
    auto modelName = getModelNames(*fixture.reader)[0];
    std::string sampleInputsPath = fmt::format(
        torch::_export::archive_spec::SAMPLE_INPUTS_FILENAME_FORMAT, modelName);

    std::vector<c10::IValue> inputs;
    if (fixture.reader->hasRecord(sampleInputsPath)) {
      auto size = fixture.reader->getRecordSize(sampleInputsPath);
      std::vector<char> buffers(size);
      fixture.reader->getRecord(sampleInputsPath, buffers.data(), size);
      auto value = torch::jit::pickle_load(buffers);
      if (value.isTuple() && value.toTupleRef().elements().size() == 2) {
        const auto& argsVal = value.toTupleRef().elements().at(0);
        if (argsVal.isTuple()) {
          for (const auto& arg : argsVal.toTupleRef().elements()) {
            inputs.push_back(arg);
          }
        }
        const auto& kwargsVal = value.toTupleRef().elements().at(1);
        if (kwargsVal.isTuple()) {
          for (const auto& kwarg : kwargsVal.toTupleRef().elements()) {
            inputs.push_back(kwarg);
          }
        } else if (kwargsVal.isGenericDict()) {
          for (const auto& entry : kwargsVal.toGenericDict()) {
            inputs.push_back(entry.value());
          }
        }
      }
    }
    EXPECT_FALSE(inputs.empty()) << "No sample inputs found in package";
    return inputs;
  }

  /// Verifies that IValue outputs match expected tensors.
  void verifyOutputs(
      const std::vector<c10::IValue>& outputs,
      const std::vector<at::Tensor>& expected,
      const std::string& label) {
    ASSERT_EQ(outputs.size(), expected.size())
        << label << ": output count mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_TRUE(outputs[i].isTensor())
          << label << " output " << i << " is not a tensor";
      EXPECT_TRUE(outputs[i].toTensor().equal(expected[i]))
          << label << " output " << i << " differs from expected"
          << "\n  expected: " << tensorDebugString(expected[i])
          << "\n  actual:   " << tensorDebugString(outputs[i].toTensor());
    }
  }

  /// Verifies that tensor outputs match expected tensors.
  void verifyOutputs(
      const std::vector<at::Tensor>& outputs,
      const std::vector<at::Tensor>& expected,
      const std::string& label) {
    ASSERT_EQ(outputs.size(), expected.size())
        << label << ": output count mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_TRUE(outputs[i].equal(expected[i]))
          << label << " output " << i << " differs from expected"
          << "\n  expected: " << tensorDebugString(expected[i])
          << "\n  actual:   " << tensorDebugString(outputs[i]);
    }
  }
};

TEST_F(ExecutorTest, elementTest) {
  runTest("data/element_test.pt2", "data/element_test_results.pt");
}

TEST_F(ExecutorTest, maskedSelectTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/masked_select_test.pt2",
      "data/masked_select_test_results.pt",
      "single");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/masked_select_test.pt2",
      "data/masked_select_test_results.pt",
      "3 step");
}

TEST_F(ExecutorTest, custom) {
  if (FLAGS_custom.empty()) {
    return;
  }
  runTest(FLAGS_custom + ".pt2", FLAGS_custom + "_results.pt");
}

} // namespace
} // namespace torch::wave

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv};
  if (int device; cudaGetDevice(&device) != cudaSuccess) {
    LOG(WARNING) << "No CUDA detected, skipping all tests";
    return 0;
  }
  return RUN_ALL_TESTS();
}
