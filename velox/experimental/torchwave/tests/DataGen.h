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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::wave {

/// Where a leaf value comes from in the graph. User inputs are supplied per
/// run; the rest are persistent weights (Weights, keyed by name).
enum class LeafKind {
  kUserInput = 0, ///< A user-supplied graph input tensor.
  kUserInputScalar =
      1, ///< A user-supplied graph input scalar (int/float/bool).
  kParameter = 2, ///< A model parameter (Weights).
  kBuffer = 3, ///< A model buffer (Weights).
  kConstant = 4, ///< A tensor constant (Weights).
};

/// Statistical description of one leaf tensor (or scalar) of a model, produced
/// by analyzing a real dataset. It captures enough to regenerate synthetic data
/// in the same general range and with a similar number of distinct values,
/// without keeping the original (possibly private) data. Small tensors and
/// detected offset arrays keep their exact values.
struct LeafSpec {
  LeafKind kind{LeafKind::kUserInput};

  /// Graph value id (informational; weights are matched by name, user inputs by
  /// position).
  int32_t valueId{-1};
  /// Weight FQN (empty for user inputs).
  std::string name;
  /// Position in graph.userInputs() (only for user inputs; -1 otherwise).
  int32_t inputIndex{-1};

  /// c10::ScalarType name (e.g. "Float", "Long"). Empty for non-tensor scalars,
  /// which use 'scalarValue' instead.
  std::string dtype;
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;

  double minVal{0};
  double maxVal{0};
  int64_t distinctCount{0};
  double sum{0};

  /// True when this int/long tensor looks like a jagged offsets array
  /// (ascending, last element within 5% of some other leaf's element count).
  bool isOffsets{false};

  /// When true, 'values' lists every element (offsets, or small tensors); the
  /// generator reproduces them exactly.
  bool hasValues{false};
  std::vector<double> values;

  /// For a scalar user input (kUserInputScalar): the exact value, reproduced
  /// as-is.
  double scalarValue{0};
  /// "int" | "float" | "bool" for a scalar user input.
  std::string scalarType;
};

/// A full dataset specification: one entry per leaf value of the model.
struct DatasetSpec {
  std::vector<LeafSpec> leaves;
  /// Optional seed used at generation time (0 means unset / caller-provided).
  uint64_t seed{0};
};

/// Analyzes the real leaf values of a model and writes a DatasetSpec as JSON to
/// 'specPath'. 'userInputs' are the actual user inputs (positional, matching
/// graph.userInputs()); the weights are read from 'weights'. This is the
/// analysis half of the synthesizer -- its input is the actual dataset, its
/// output is a per-tensor specification.
void makeDatasetSpec(
    const nativert::Graph& graph,
    const nativert::Weights& weights,
    const std::vector<c10::IValue>& userInputs,
    const std::string& specPath);

/// Parses a DatasetSpec JSON file (does not generate any tensors).
DatasetSpec loadDatasetSpec(const std::string& specPath);

/// Result of generating synthetic data from a DatasetSpec: freshly built
/// Weights bound to 'graph' plus the positional user inputs (CPU tensors /
/// scalars).
struct GeneratedData {
  std::shared_ptr<nativert::Weights> weights;
  std::vector<c10::IValue> userInputs;
};

/// Reads the spec at 'specPath' and generates pseudorandom synthetic data for
/// 'graph': a Weights object with all parameters/buffers/constants filled, and
/// the positional user inputs. Values are drawn to match each leaf's recorded
/// range and distinct-value count; offsets and small tensors are reproduced
/// exactly. If 'seed' is set it overrides the spec's seed; generation is
/// deterministic for a given seed. User-input tensors are created on CPU;
/// weights are placed on 'weightDevice' (default CPU) so a GPU reference run
/// gets device-resident weights consistent with a device-placed graph.
GeneratedData generateFromSpec(
    const nativert::Graph& graph,
    const std::string& specPath,
    std::optional<uint64_t> seed = std::nullopt,
    c10::Device weightDevice = c10::kCPU);

} // namespace torch::wave
