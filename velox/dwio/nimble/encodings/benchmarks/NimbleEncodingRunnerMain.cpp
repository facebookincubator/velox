/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/benchmarks/NimbleEncodingRunner.h"

DEFINE_string(task_id, "", "Versioned Nimble benchmark task ID");
DEFINE_string(mode, "sample", "Runner mode: sample or verify");
DEFINE_string(artifact_path, "", "Artifact output or verification input path");
DEFINE_string(
    input_artifact_path,
    "",
    "Canonical artifact input for a consumer-lane sample");
DEFINE_uint32(row_count, 4096, "Deterministic corpus row count");
DEFINE_uint64(seed, 0xC0FFEE, "Deterministic corpus seed");
DEFINE_uint32(warmups, 3, "Untimed warmup samples");
DEFINE_uint32(samples, 15, "Timed samples");
DEFINE_uint32(
    min_sample_time_micros,
    250'000,
    "Minimum duration of each timed sample");
DEFINE_uint32(inner_iterations, 1, "Initial operations per timed sample");

namespace {

using facebook::nimble::benchmarks::EncodingRunnerConfig;

EncodingRunnerConfig runnerConfig() {
  return EncodingRunnerConfig{
      .taskId = FLAGS_task_id,
      .rowCount = FLAGS_row_count,
      .seed = FLAGS_seed,
      .warmups = FLAGS_warmups,
      .samples = FLAGS_samples,
      .minSampleTimeMicros = FLAGS_min_sample_time_micros,
      .innerIterations = FLAGS_inner_iterations,
  };
}

void writeArtifact(const std::string& path, std::string_view artifact) {
  std::ofstream output{path, std::ios::binary | std::ios::trunc};
  if (!output) {
    throw std::runtime_error("Unable to open artifact output: " + path);
  }
  output.write(artifact.data(), artifact.size());
  if (!output) {
    throw std::runtime_error("Unable to write artifact output: " + path);
  }
}

std::string readArtifact(const std::string& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error("Unable to open artifact input: " + path);
  }
  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  if (end < 0) {
    throw std::runtime_error("Unable to determine artifact size: " + path);
  }
  const auto size = static_cast<size_t>(end);
  if (size > facebook::nimble::benchmarks::kMaxEncodingArtifactBytes) {
    throw std::runtime_error("Artifact exceeds the runner size limit: " + path);
  }

  std::string artifact(size, '\0');
  input.seekg(0, std::ios::beg);
  if (!input) {
    throw std::runtime_error("Unable to seek artifact input: " + path);
  }
  if (!artifact.empty()) {
    input.read(artifact.data(), static_cast<std::streamsize>(artifact.size()));
    if (!input || static_cast<size_t>(input.gcount()) != artifact.size()) {
      throw std::runtime_error("Unable to read artifact input: " + path);
    }
  }
  char extra;
  if (input.get(extra)) {
    throw std::runtime_error("Artifact changed while being read: " + path);
  }
  return artifact;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  try {
    facebook::velox::memory::MemoryManager::initialize({});
    auto pool = facebook::velox::memory::memoryManager()->addLeafPool(
        "nimble_encoding_runner");
    const auto config = runnerConfig();

    if (FLAGS_mode == "sample") {
      const auto inputArtifact = FLAGS_input_artifact_path.empty()
          ? std::nullopt
          : std::optional<std::string>{readArtifact(FLAGS_input_artifact_path)};
      const auto inputArtifactView = inputArtifact.has_value()
          ? std::optional<std::string_view>{*inputArtifact}
          : std::nullopt;
      auto measurement = facebook::nimble::benchmarks::runEncodingBenchmark(
          config, *pool, inputArtifactView);
      if (!FLAGS_artifact_path.empty()) {
        writeArtifact(FLAGS_artifact_path, measurement.encodedArtifact);
      }
      std::cout << facebook::nimble::benchmarks::measurementToJson(measurement)
                << '\n';
      return 0;
    }

    if (FLAGS_mode == "verify") {
      if (!FLAGS_input_artifact_path.empty()) {
        throw std::invalid_argument(
            "--input_artifact_path is only valid in sample mode");
      }
      if (FLAGS_artifact_path.empty()) {
        throw std::invalid_argument(
            "--artifact_path is required in verify mode");
      }
      const auto verification =
          facebook::nimble::benchmarks::verifyEncodingArtifact(
              config, readArtifact(FLAGS_artifact_path), *pool);
      std::cout << facebook::nimble::benchmarks::verificationToJson(
                       verification)
                << '\n';
      return 0;
    }

    throw std::invalid_argument("--mode must be sample or verify");
  } catch (const std::exception& error) {
    std::cerr << "Nimble encoding runner failed: " << error.what() << '\n';
    return 1;
  }
}
