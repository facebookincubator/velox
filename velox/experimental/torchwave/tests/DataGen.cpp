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

#include "velox/experimental/torchwave/tests/DataGen.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <c10/core/ScalarType.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <glog/logging.h>

#include <torch/nativert/graph/GraphSignature.h>

namespace torch::wave {
namespace {

// Cap for recording all values of a small tensor and for the distinct-value
// pool used at generation time. Tensors below this element count keep their
// exact values in the spec.
constexpr int64_t kListValuesCap = 256;

// Cap on the distinct-value set built during analysis (bounds memory) and on
// the pool used at generation time.
constexpr int64_t kDistinctCap = 1 << 20;

// Fraction by which an ascending int tensor's last value may differ from a
// candidate tensor's element count and still be treated as offsets.
constexpr double kOffsetsTolerance = 0.05;

std::string readWholeFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  TORCH_CHECK(in.is_open(), "Cannot open spec file: ", path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void writeWholeFile(const std::string& path, const std::string& contents) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  TORCH_CHECK(out.is_open(), "Cannot write spec file: ", path);
  out << contents;
}

c10::ScalarType scalarTypeFromString(const std::string& name) {
  static const std::unordered_map<std::string, c10::ScalarType> kMap = {
      {"Float", at::kFloat},
      {"Double", at::kDouble},
      {"Half", at::kHalf},
      {"BFloat16", at::kBFloat16},
      {"Long", at::kLong},
      {"Int", at::kInt},
      {"Short", at::kShort},
      {"Char", at::kChar},
      {"Byte", at::kByte},
      {"Bool", at::kBool},
  };
  auto it = kMap.find(name);
  if (it != kMap.end()) {
    return it->second;
  }
  LOG(WARNING) << "DataGen: unknown dtype '" << name
               << "', defaulting to Float";
  return at::kFloat;
}

int64_t product(const std::vector<int64_t>& dims) {
  int64_t n = 1;
  for (auto d : dims) {
    n *= d;
  }
  return n;
}

// Fills the numeric statistics (min/max/sum/distinctCount) and, when the tensor
// is small or looks like offsets, the exact values, into 'spec'. 'allNumels'
// holds every leaf tensor's element count for the offsets heuristic.
void analyzeTensor(
    const at::Tensor& tensor,
    const std::vector<int64_t>& allNumels,
    LeafSpec& spec) {
  spec.dtype = c10::toString(tensor.scalar_type());
  spec.dims = tensor.sizes().vec();
  spec.strides = tensor.strides().vec();

  auto cpu = tensor.to(at::kCPU).contiguous();
  int64_t n = cpu.numel();
  if (n == 0) {
    return;
  }

  bool isInt = c10::isIntegralType(tensor.scalar_type(), /*includeBool=*/false);

  auto asDouble = cpu.to(at::kDouble).contiguous();
  const double* p = asDouble.data_ptr<double>();
  double minVal = p[0];
  double maxVal = p[0];
  double sum = 0;
  bool ascending = true;
  std::unordered_set<double> distinct;
  for (int64_t i = 0; i < n; ++i) {
    double v = p[i];
    minVal = std::min(minVal, v);
    maxVal = std::max(maxVal, v);
    sum += v;
    if (i > 0 && v < p[i - 1]) {
      ascending = false;
    }
    if (static_cast<int64_t>(distinct.size()) < kDistinctCap) {
      distinct.insert(v);
    }
  }
  spec.minVal = minVal;
  spec.maxVal = maxVal;
  spec.sum = sum;
  spec.distinctCount = static_cast<int64_t>(distinct.size());

  // Offsets heuristic: an ascending int tensor whose last element is within a
  // few percent of some leaf's element count is a jagged offsets array.
  if (isInt && n > 1 && ascending) {
    double last = p[n - 1];
    for (int64_t m : allNumels) {
      if (m <= 0) {
        continue;
      }
      if (std::abs(last - static_cast<double>(m)) <=
          kOffsetsTolerance * static_cast<double>(m)) {
        spec.isOffsets = true;
        break;
      }
    }
  }

  // Lengths heuristic: an int tensor whose element sum is within a few percent
  // of some leaf's element count is a jagged lengths array (sum of lengths ==
  // number of values). Sparse ops (e.g. batch_flip_and_truncate /
  // GroupLengthGuardSparse) memcpy each group by length, so an inconsistent
  // lengths/values pair reads out of bounds. Record such tensors exactly so the
  // sum-of-lengths invariant -- and the resulting data cardinalities -- are
  // preserved. Feature-value tensors (hashes) have astronomically large sums
  // that match no size, so they are never mistaken for lengths.
  bool isLengths = false;
  if (isInt && !spec.isOffsets) {
    for (int64_t m : allNumels) {
      if (m <= 0) {
        continue;
      }
      if (std::abs(sum - static_cast<double>(m)) <=
          kOffsetsTolerance * static_cast<double>(m)) {
        isLengths = true;
        break;
      }
    }
  }

  // Record exact values for offsets, lengths, and small tensors so generation
  // reproduces their structure.
  if (spec.isOffsets || isLengths || n <= kListValuesCap) {
    spec.hasValues = true;
    spec.values.assign(p, p + n);
  }
}

LeafSpec analyzeScalar(const c10::IValue& iv) {
  LeafSpec spec;
  spec.kind = LeafKind::kUserInputScalar;
  if (iv.isInt()) {
    spec.scalarType = "int";
    spec.scalarValue = static_cast<double>(iv.toInt());
  } else if (iv.isDouble()) {
    spec.scalarType = "float";
    spec.scalarValue = iv.toDouble();
  } else if (iv.isBool()) {
    spec.scalarType = "bool";
    spec.scalarValue = iv.toBool() ? 1.0 : 0.0;
  } else {
    // None or an unsupported input type: reproduce as None.
    spec.scalarType = "none";
  }
  return spec;
}

folly::dynamic leafToDynamic(const LeafSpec& leaf) {
  folly::dynamic o = folly::dynamic::object;
  o["kind"] = static_cast<int>(leaf.kind);
  o["valueId"] = leaf.valueId;
  o["name"] = leaf.name;
  o["inputIndex"] = leaf.inputIndex;
  o["dtype"] = leaf.dtype;
  folly::dynamic dims = folly::dynamic::array;
  for (auto d : leaf.dims) {
    dims.push_back(d);
  }
  o["dims"] = dims;
  folly::dynamic strides = folly::dynamic::array;
  for (auto s : leaf.strides) {
    strides.push_back(s);
  }
  o["strides"] = strides;
  o["min"] = leaf.minVal;
  o["max"] = leaf.maxVal;
  o["distinctCount"] = leaf.distinctCount;
  o["sum"] = leaf.sum;
  o["isOffsets"] = leaf.isOffsets;
  o["hasValues"] = leaf.hasValues;
  if (leaf.hasValues) {
    folly::dynamic values = folly::dynamic::array;
    for (auto v : leaf.values) {
      values.push_back(v);
    }
    o["values"] = values;
  }
  if (leaf.kind == LeafKind::kUserInputScalar) {
    o["scalarType"] = leaf.scalarType;
    o["scalarValue"] = leaf.scalarValue;
  }
  return o;
}

LeafSpec leafFromDynamic(const folly::dynamic& o) {
  LeafSpec leaf;
  leaf.kind = static_cast<LeafKind>(o["kind"].asInt());
  leaf.valueId = static_cast<int32_t>(o["valueId"].asInt());
  leaf.name = o["name"].asString();
  leaf.inputIndex = static_cast<int32_t>(o["inputIndex"].asInt());
  leaf.dtype = o["dtype"].asString();
  for (const auto& d : o["dims"]) {
    leaf.dims.push_back(d.asInt());
  }
  for (const auto& s : o["strides"]) {
    leaf.strides.push_back(s.asInt());
  }
  leaf.minVal = o["min"].asDouble();
  leaf.maxVal = o["max"].asDouble();
  leaf.distinctCount = o["distinctCount"].asInt();
  leaf.sum = o["sum"].asDouble();
  leaf.isOffsets = o["isOffsets"].asBool();
  leaf.hasValues = o["hasValues"].asBool();
  if (leaf.hasValues && o.count("values")) {
    for (const auto& v : o["values"]) {
      leaf.values.push_back(v.asDouble());
    }
  }
  if (leaf.kind == LeafKind::kUserInputScalar) {
    leaf.scalarType = o["scalarType"].asString();
    leaf.scalarValue = o["scalarValue"].asDouble();
  }
  return leaf;
}

// Builds a CPU tensor for one leaf spec. Uses the global RNG (seed set by the
// caller). Reproduces exact values when recorded, otherwise draws random values
// matching the recorded range and distinct-value count.
at::Tensor generateTensor(const LeafSpec& leaf) {
  auto stype = scalarTypeFromString(leaf.dtype);
  auto opts = at::TensorOptions().dtype(stype).device(at::kCPU);
  int64_t n = product(leaf.dims);

  if (leaf.hasValues) {
    auto flat =
        at::empty({static_cast<int64_t>(leaf.values.size())}, at::kDouble);
    double* d = flat.data_ptr<double>();
    std::copy(leaf.values.begin(), leaf.values.end(), d);
    if (static_cast<int64_t>(leaf.values.size()) != n) {
      // Fall back to the recorded 1-D values if they do not match dims.
      return flat.to(stype);
    }
    return flat.reshape(leaf.dims).to(stype);
  }

  if (n == 0) {
    return at::empty(leaf.dims, opts);
  }

  bool isInt = c10::isIntegralType(stype, /*includeBool=*/false);
  bool usePool = leaf.distinctCount > 0 && leaf.distinctCount <= kDistinctCap &&
      leaf.distinctCount < n;

  if (stype == at::kBool) {
    return at::randint(0, 2, leaf.dims, opts);
  }

  if (usePool) {
    int64_t poolSize = leaf.distinctCount;
    at::Tensor pool;
    if (isInt) {
      int64_t lo = static_cast<int64_t>(std::llround(leaf.minVal));
      int64_t hi = static_cast<int64_t>(std::llround(leaf.maxVal));
      pool = at::randint(lo, hi + 1, {poolSize}, opts);
    } else {
      pool = at::rand({poolSize}, at::TensorOptions().dtype(stype)) *
              (leaf.maxVal - leaf.minVal) +
          leaf.minVal;
    }
    auto idx = at::randint(
        0,
        poolSize,
        {n},
        at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    return pool.index_select(0, idx).reshape(leaf.dims);
  }

  if (isInt) {
    int64_t lo = static_cast<int64_t>(std::llround(leaf.minVal));
    int64_t hi = static_cast<int64_t>(std::llround(leaf.maxVal));
    return at::randint(lo, hi + 1, leaf.dims, opts);
  }

  return (at::rand(leaf.dims, at::TensorOptions().dtype(stype)) *
              (leaf.maxVal - leaf.minVal) +
          leaf.minVal)
      .to(stype);
}

c10::IValue generateScalar(const LeafSpec& leaf) {
  if (leaf.scalarType == "int") {
    return c10::IValue(static_cast<int64_t>(std::llround(leaf.scalarValue)));
  }
  if (leaf.scalarType == "float") {
    return c10::IValue(leaf.scalarValue);
  }
  if (leaf.scalarType == "bool") {
    return c10::IValue(leaf.scalarValue != 0.0);
  }
  return c10::IValue();
}

} // namespace

void makeDatasetSpec(
    const nativert::Graph& graph,
    const nativert::Weights& weights,
    const std::vector<c10::IValue>& userInputs,
    const std::string& specPath) {
  DatasetSpec spec;

  // Collect every leaf tensor's element count first (needed for the offsets
  // heuristic), then analyze.
  std::vector<int64_t> allNumels;
  for (const auto& iv : userInputs) {
    if (iv.isTensor()) {
      allNumels.push_back(iv.toTensor().numel());
    }
  }
  auto collectNumels =
      [&](const std::unordered_map<std::string, at::Tensor>& m) {
        for (const auto& [name, t] : m) {
          allNumels.push_back(t.numel());
        }
      };
  auto params = weights.parameters();
  auto buffers = weights.buffers();
  auto constants = weights.attributes();
  collectNumels(params);
  collectNumels(buffers);
  collectNumels(constants);

  // Map weight FQN -> graph value id (best effort; informational only).
  std::unordered_map<std::string, int32_t> fqnToId;
  {
    std::unordered_map<std::string, int32_t> inputNameToId;
    for (const auto* v : graph.weightValues()) {
      inputNameToId[std::string(v->name())] = v->id();
    }
    for (const auto& [inputName, fqn] : graph.signature().inputsToWeights()) {
      auto it = inputNameToId.find(inputName);
      if (it != inputNameToId.end()) {
        fqnToId[fqn] = it->second;
      }
    }
  }

  // User inputs (positional, matching graph.userInputs()).
  const auto& inputValues = graph.userInputs();
  const auto& inputNames = graph.signature().userInputs();
  for (size_t i = 0; i < userInputs.size(); ++i) {
    const auto& iv = userInputs[i];
    if (iv.isTensor()) {
      LeafSpec leaf;
      leaf.kind = LeafKind::kUserInput;
      leaf.inputIndex = static_cast<int32_t>(i);
      if (i < inputValues.size() && inputValues[i]) {
        leaf.valueId = inputValues[i]->id();
      }
      if (i < inputNames.size()) {
        leaf.name = inputNames[i];
      }
      analyzeTensor(iv.toTensor(), allNumels, leaf);
      spec.leaves.push_back(std::move(leaf));
    } else {
      LeafSpec leaf = analyzeScalar(iv);
      leaf.inputIndex = static_cast<int32_t>(i);
      if (i < inputValues.size() && inputValues[i]) {
        leaf.valueId = inputValues[i]->id();
      }
      if (i < inputNames.size()) {
        leaf.name = inputNames[i];
      }
      spec.leaves.push_back(std::move(leaf));
    }
  }

  // Weights: parameters, buffers, constants (keyed by FQN).
  auto addWeights = [&](const std::unordered_map<std::string, at::Tensor>& m,
                        LeafKind kind) {
    for (const auto& [name, t] : m) {
      LeafSpec leaf;
      leaf.kind = kind;
      leaf.name = name;
      auto it = fqnToId.find(name);
      leaf.valueId = it != fqnToId.end() ? it->second : -1;
      analyzeTensor(t, allNumels, leaf);
      spec.leaves.push_back(std::move(leaf));
    }
  };
  addWeights(params, LeafKind::kParameter);
  addWeights(buffers, LeafKind::kBuffer);
  addWeights(constants, LeafKind::kConstant);

  folly::dynamic root = folly::dynamic::object;
  root["seed"] = static_cast<int64_t>(spec.seed);
  folly::dynamic arr = folly::dynamic::array;
  for (const auto& leaf : spec.leaves) {
    arr.push_back(leafToDynamic(leaf));
  }
  root["leaves"] = arr;
  writeWholeFile(specPath, folly::toPrettyJson(root));
  LOG(INFO) << "DataGen: wrote spec with " << spec.leaves.size()
            << " leaves to " << specPath;
}

DatasetSpec loadDatasetSpec(const std::string& specPath) {
  auto contents = readWholeFile(specPath);
  auto root = folly::parseJson(contents);
  DatasetSpec spec;
  if (root.count("seed")) {
    spec.seed = static_cast<uint64_t>(root["seed"].asInt());
  }
  for (const auto& o : root["leaves"]) {
    spec.leaves.push_back(leafFromDynamic(o));
  }
  return spec;
}

GeneratedData generateFromSpec(
    const nativert::Graph& graph,
    const std::string& specPath,
    std::optional<uint64_t> seed,
    c10::Device weightDevice) {
  DatasetSpec spec = loadDatasetSpec(specPath);
  uint64_t actualSeed = seed.has_value() ? *seed : spec.seed;
  at::manual_seed(actualSeed);

  GeneratedData out;

  // Count user inputs to size the positional output vector.
  int32_t maxInputIndex = -1;
  for (const auto& leaf : spec.leaves) {
    if (leaf.kind == LeafKind::kUserInput ||
        leaf.kind == LeafKind::kUserInputScalar) {
      maxInputIndex = std::max(maxInputIndex, leaf.inputIndex);
    }
  }
  out.userInputs.resize(maxInputIndex + 1);

  std::unordered_map<std::string, c10::IValue> stateDict;
  std::unordered_map<std::string, c10::IValue> constants;

  // A weight's shape and dtype must match the graph's weightsMeta exactly (the
  // Weights ctor validates them), so generate weights to the meta rather than
  // the recorded spec dims in case the two ever diverge.
  const auto& weightsMeta = graph.weightsMeta();
  auto generateWeight = [&](const LeafSpec& leaf) -> at::Tensor {
    at::Tensor t;
    auto it = weightsMeta.find(leaf.name);
    if (it == weightsMeta.end() || it->second.hasSymbolicShape()) {
      t = generateTensor(leaf);
    } else {
      LeafSpec adjusted = leaf;
      adjusted.dims = it->second.sizes().vec();
      adjusted.dtype = c10::toString(it->second.dtype());
      if (adjusted.hasValues &&
          static_cast<int64_t>(adjusted.values.size()) !=
              product(adjusted.dims)) {
        adjusted.hasValues = false;
        adjusted.values.clear();
      }
      t = generateTensor(adjusted);
    }
    return weightDevice.is_cpu() ? t : t.to(weightDevice);
  };

  for (const auto& leaf : spec.leaves) {
    switch (leaf.kind) {
      case LeafKind::kUserInput:
        if (leaf.inputIndex >= 0) {
          out.userInputs[leaf.inputIndex] = generateTensor(leaf);
        }
        break;
      case LeafKind::kUserInputScalar:
        if (leaf.inputIndex >= 0) {
          out.userInputs[leaf.inputIndex] = generateScalar(leaf);
        }
        break;
      case LeafKind::kParameter:
      case LeafKind::kBuffer:
        stateDict[leaf.name] = generateWeight(leaf);
        break;
      case LeafKind::kConstant:
        constants[leaf.name] = generateWeight(leaf);
        break;
    }
  }

  out.weights = std::make_shared<nativert::Weights>(
      &graph,
      stateDict.empty()
          ? std::nullopt
          : std::optional<std::unordered_map<std::string, c10::IValue>>(
                std::move(stateDict)),
      constants.empty()
          ? std::nullopt
          : std::optional<std::unordered_map<std::string, c10::IValue>>(
                std::move(constants)));
  LOG(INFO) << "DataGen: generated data for " << spec.leaves.size()
            << " leaves (seed " << actualSeed << ")";
  return out;
}

} // namespace torch::wave
