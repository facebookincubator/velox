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

#include <chrono>
#include <fstream>

#include <ATen/ATen.h>
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

#ifndef ELEMENT_TEST_PT2_PATH
#define ELEMENT_TEST_PT2_PATH "/tmp/element_test.pt2"
#endif

#ifndef ELEMENT_TEST_RESULTS_PATH
#define ELEMENT_TEST_RESULTS_PATH "/tmp/element_test_results.pt"
#endif

namespace torch::wave {
namespace {

using Clock = std::chrono::high_resolution_clock;

/// Timing results from running a model through an executor.
struct RunTiming {
  /// Time to transfer inputs to device (0 for CPU executors).
  int64_t dataTransferUs{0};
  /// Execution time excluding data transfer.
  int64_t executeUs{0};
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

class ExecutorTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
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

    verifyOutputs(outputs, expected, "serial");
    return {0, executeUs};
  }

  /// Runs a .pt2 model through the nativert SerialGraphExecutor on device.
  /// ATen kernels dispatch to CUDA when given device tensors.
  /// Verifies outputs match 'expected' and returns timing.
  RunTiming runSerialOnDevice(
      ModelFixture& fixture,
      const std::vector<at::Tensor>& expected) {
    const auto& graph = *fixture.model.graph;

    auto kernels = fixture.makeKernels();
    nativert::SerialGraphExecutor executor(
        graph, std::move(kernels), fixture.config);

    auto frame = std::make_unique<nativert::ExecutionFrame>(
        graph, *fixture.weights, fixture.config);

    // Load sample inputs and move to device.
    auto inputs = loadSampleInputs(fixture);
    std::vector<at::Tensor> inputTensors;
    for (auto& iv : inputs) {
      if (iv.isTensor()) {
        inputTensors.push_back(iv.toTensor());
      }
    }

    facebook::velox::wave::Stream stream;
    std::vector<at::Tensor> deviceInputs;

    auto dataMovStart = Clock::now();
    tensorsToDevice(inputTensors, deviceInputs, stream);
    stream.wait();
    auto dataMovUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - dataMovStart)
                         .count();

    // Build IValue inputs from device tensors.
    std::vector<c10::IValue> deviceIvalues;
    deviceIvalues.reserve(deviceInputs.size());
    for (auto& t : deviceInputs) {
      deviceIvalues.emplace_back(std::move(t));
    }

    auto start = Clock::now();
    auto outputs = executor.execute(*frame, std::move(deviceIvalues));
    auto executeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - start)
                         .count();

    // Copy outputs to host for comparison.
    std::vector<at::Tensor> deviceOutputTensors;
    for (auto& iv : outputs) {
      EXPECT_TRUE(iv.isTensor()) << "Serial GPU output is not a tensor";
      if (iv.isTensor()) {
        deviceOutputTensors.push_back(iv.toTensor());
      }
    }

    facebook::velox::wave::Stream hostStream;
    std::vector<at::Tensor> hostOutputs;
    tensorsToHost(deviceOutputTensors, hostOutputs, hostStream);
    hostStream.wait();

    verifyOutputs(hostOutputs, expected, "serial-gpu");
    return {dataMovUs, executeUs};
  }

  /// Runs a .pt2 model through the WaveGraphExecutor on device.
  /// Verifies outputs match 'expected' and returns timing.
  RunTiming runWave(
      ModelFixture& fixture,
      const std::vector<at::Tensor>& expected) {
    const auto& graph = *fixture.model.graph;

    WaveGraphExecutor waveExec(
        graph, fixture.makeKernels(), fixture.config, fixture.weights);

    // Get a device frame (not timed).
    auto pooledFrame = waveExec.getFrame();
    EXPECT_NE(pooledFrame, nullptr);

    // Load sample inputs and move to device (timed).
    auto inputs = loadSampleInputs(fixture);
    std::vector<at::Tensor> inputTensors;
    for (auto& iv : inputs) {
      if (iv.isTensor()) {
        inputTensors.push_back(iv.toTensor());
      }
    }

    facebook::velox::wave::Stream stream;
    std::vector<at::Tensor> deviceInputs;

    auto dataMovStart = Clock::now();
    tensorsToDevice(inputTensors, deviceInputs, stream);
    stream.wait();
    auto dataMovUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - dataMovStart)
                         .count();

    // Fill device inputs into the frame.
    const auto& userInputNames = graph.signature().userInputs();
    size_t tensorIdx = 0;
    for (const auto& name : userInputNames) {
      auto* value = graph.tryGetValue(name);
      EXPECT_NE(value, nullptr) << "No value for input " << name;
      if (value && tensorIdx < deviceInputs.size()) {
        pooledFrame->setIValue(
            value->id(), c10::IValue(std::move(deviceInputs[tensorIdx++])));
      }
    }

    // Time wave executor (excluding frame acquisition and data transfer).
    auto waveStart = Clock::now();
    auto waveOutputs = waveExec.executeWithPrefilledFrame(*pooledFrame);
    auto waveUs = std::chrono::duration_cast<std::chrono::microseconds>(
                      Clock::now() - waveStart)
                      .count();

    waveExec.returnFrame(std::move(pooledFrame));

    // Copy outputs to host for comparison.
    std::vector<at::Tensor> deviceOutputTensors;
    for (auto& iv : waveOutputs) {
      EXPECT_TRUE(iv.isTensor()) << "Wave output is not a tensor";
      if (iv.isTensor()) {
        deviceOutputTensors.push_back(iv.toTensor());
      }
    }

    facebook::velox::wave::Stream hostStream;
    std::vector<at::Tensor> hostOutputs;
    tensorsToHost(deviceOutputTensors, hostOutputs, hostStream);
    hostStream.wait();

    verifyOutputs(hostOutputs, expected, "wave");
    return {dataMovUs, waveUs};
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
          << label << " output " << i << " differs from expected";
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
          << label << " output " << i << " differs from expected";
    }
  }
};

TEST_F(ExecutorTest, elementTest) {
  auto fixture = ModelFixture::load(ELEMENT_TEST_PT2_PATH);
  ASSERT_NE(fixture, nullptr);

  auto expected = loadReferenceTensors(ELEMENT_TEST_RESULTS_PATH);
  ASSERT_FALSE(expected.empty());

  auto serialTiming = runSerial(*fixture, expected);
  auto serialGpuTiming = runSerialOnDevice(*fixture, expected);
  auto waveTiming = runWave(*fixture, expected);

  LOG(INFO) << "element_test serial CPU: " << serialTiming.executeUs << " us";
  LOG(INFO) << "element_test serial GPU data transfer (H2D): "
            << serialGpuTiming.dataTransferUs << " us";
  LOG(INFO) << "element_test serial GPU executor: " << serialGpuTiming.executeUs
            << " us";
  LOG(INFO) << "element_test wave data transfer (H2D): "
            << waveTiming.dataTransferUs << " us";
  LOG(INFO) << "element_test wave executor: " << waveTiming.executeUs << " us";
}

} // namespace
} // namespace torch::wave
