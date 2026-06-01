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

#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/WaveConfig.h"

#include <type_traits>

#include <fstream>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/StringUtil.h>
#include <fmt/format.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/Placement.h>

namespace torch::wave {

std::vector<ValueCP> inputValues(NodeCP node) {
  std::vector<ValueCP> result;
  result.reserve(node->inputs().size());
  for (const auto& input : node->inputs()) {
    result.push_back(input.value);
  }
  return result;
}

std::unordered_set<ValueCP> inputValueSet(NodeCP node) {
  std::unordered_set<ValueCP> result;
  result.reserve(node->inputs().size());
  for (const auto& input : node->inputs()) {
    result.insert(input.value);
  }
  return result;
}

const c10::FunctionSchema* findFunctionSchema(std::string_view qualifiedName) {
  auto atoms = c10::split(qualifiedName, '.');
  if (atoms.size() < 3) {
    return nullptr;
  }
  auto numAtoms = atoms.size();
  auto ns = atoms[numAtoms - 3];
  auto opName = atoms[numAtoms - 2];
  auto overloadName = atoms[numAtoms - 1];
  auto operatorName = fmt::format("{}::{}", ns, opName);
  std::string normalizedOverload =
      (overloadName == "default") ? "" : std::string(overloadName);
  auto op = c10::Dispatcher::singleton().findSchema(
      {operatorName.c_str(), normalizedOverload.c_str()});
  if (!op.has_value()) {
    return nullptr;
  }
  return &op->schema();
}

std::string constantToString(const nativert::Constant& c) {
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, nativert::None>) {
          return "None";
        } else if constexpr (std::is_same_v<T, bool>) {
          return v ? "True" : "False";
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return std::to_string(v);
        } else if constexpr (std::is_same_v<T, double>) {
          return fmt::format("{}", v);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return fmt::format("\"{}\"", v);
        } else if constexpr (std::is_same_v<T, c10::ScalarType>) {
          return c10::toString(v);
        } else if constexpr (std::is_same_v<T, c10::MemoryFormat>) {
          return c10::toString(v);
        } else if constexpr (std::is_same_v<T, c10::Layout>) {
          return c10::toString(v);
        } else if constexpr (std::is_same_v<T, c10::Device>) {
          return v.str();
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          std::string r = "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) {
              r += ", ";
            }
            r += std::to_string(v[i]);
          }
          return r + "]";
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          std::string r = "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) {
              r += ", ";
            }
            r += fmt::format("{}", v[i]);
          }
          return r + "]";
        } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
          std::string r = "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) {
              r += ", ";
            }
            r += v[i] ? "True" : "False";
          }
          return r + "]";
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          std::string r = "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) {
              r += ", ";
            }
            r += fmt::format("\"{}\"", v[i]);
          }
          return r + "]";
        } else if constexpr (std::is_same_v<
                                 T,
                                 std::vector<std::vector<int64_t>>>) {
          std::string r = "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) {
              r += ", ";
            }
            r += "[";
            for (size_t j = 0; j < v[i].size(); ++j) {
              if (j > 0) {
                r += ", ";
              }
              r += std::to_string(v[i][j]);
            }
            r += "]";
          }
          return r + "]";
        } else if constexpr (std::is_same_v<
                                 T,
                                 std::unique_ptr<nativert::Graph>>) {
          return "<subgraph>";
        } else {
          return "<unknown>";
        }
      },
      c);
}

std::string ivalueToString(const c10::IValue& value) {
  if (value.isNone()) {
    return constantToString(nativert::Constant{nativert::None{}});
  }
  if (value.isBool()) {
    return constantToString(nativert::Constant{value.toBool()});
  }
  if (value.isInt()) {
    return constantToString(nativert::Constant{value.toInt()});
  }
  if (value.isDouble()) {
    return constantToString(nativert::Constant{value.toDouble()});
  }
  if (value.isString()) {
    return constantToString(
        nativert::Constant{std::string(value.toStringRef())});
  }
  if (value.isIntList()) {
    return constantToString(nativert::Constant{value.toIntVector()});
  }
  if (value.isDoubleList()) {
    return constantToString(nativert::Constant{value.toDoubleVector()});
  }
  if (value.isDevice()) {
    return constantToString(nativert::Constant{value.toDevice()});
  }
  return "<IValue>";
}

std::string cudaTypeString(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return "float";
    case c10::ScalarType::Double:
      return "double";
    case c10::ScalarType::Half:
      return "__half";
    case c10::ScalarType::BFloat16:
      return "__nv_bfloat16";
    case c10::ScalarType::Int:
      return "int32_t";
    case c10::ScalarType::Long:
      return "int64_t";
    case c10::ScalarType::Short:
      return "int16_t";
    case c10::ScalarType::Char:
      return "int8_t";
    case c10::ScalarType::Byte:
      return "uint8_t";
    case c10::ScalarType::Bool:
      return "bool";
    default:
      TORCH_CHECK(false, "Unsupported dtype ", c10::toString(dtype));
  }
}

std::string cudaTypeFromScalarTypeName(const std::string& name) {
  // Strip optional "ScalarType::" prefix.
  std::string typeName = name;
  auto pos = typeName.find("::");
  if (pos != std::string::npos) {
    typeName = typeName.substr(pos + 2);
  }
  static const std::unordered_map<std::string, std::string> kMap = {
      {"Float", "float"},
      {"Double", "double"},
      {"Half", "__half"},
      {"BFloat16", "__nv_bfloat16"},
      {"Int", "int32_t"},
      {"Long", "int64_t"},
      {"Short", "int16_t"},
      {"Char", "int8_t"},
      {"Byte", "uint8_t"},
      {"Bool", "bool"},
  };
  // PT2 export serializes dtype=None as "".  Default to Float, matching
  // PyTorch's c10::get_default_dtype() (caffe2/c10/core/DefaultDtype.cpp:5).
  if (typeName.empty()) {
    return "float";
  }
  auto it = kMap.find(typeName);
  TORCH_CHECK(it != kMap.end(), "Unsupported ScalarType name: ", name);
  return it->second;
}

std::string cudaTypeFromDtype(const nativert::Attribute& attr) {
  if (std::holds_alternative<std::string>(attr.value)) {
    return cudaTypeFromScalarTypeName(std::get<std::string>(attr.value));
  }
  if (std::holds_alternative<c10::ScalarType>(attr.value)) {
    return cudaTypeString(std::get<c10::ScalarType>(attr.value));
  }
  TORCH_CHECK(false, "dtype attribute is neither string nor ScalarType");
}

std::string dtypeName(const nativert::Attribute& attr) {
  if (std::holds_alternative<std::string>(attr.value)) {
    auto name = std::get<std::string>(attr.value);
    auto pos = name.find("::");
    return pos != std::string::npos ? name.substr(pos + 2) : name;
  }
  if (std::holds_alternative<c10::ScalarType>(attr.value)) {
    return c10::toString(std::get<c10::ScalarType>(attr.value));
  }
  TORCH_CHECK(false, "dtype attribute is neither string nor ScalarType");
}

std::string cudaTypeIdSuffix(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return "Float";
    case c10::ScalarType::Double:
      return "Double";
    case c10::ScalarType::Half:
      return "Half";
    case c10::ScalarType::BFloat16:
      return "BFloat16";
    case c10::ScalarType::Int:
      return "Int32";
    case c10::ScalarType::Long:
      return "Int64";
    case c10::ScalarType::Short:
      return "Int16";
    case c10::ScalarType::Char:
      return "Int8";
    case c10::ScalarType::Byte:
      return "UInt8";
    case c10::ScalarType::Bool:
      return "Bool";
    default:
      TORCH_CHECK(false, "Unsupported dtype ", c10::toString(dtype));
  }
}

std::string traceIValue(const c10::IValue& value) {
  if (value.isNone()) {
    return "none";
  }
  if (value.isTensor()) {
    auto& t = value.toTensor();
    std::string s = "[";
    for (int64_t d = 0; d < t.dim(); ++d) {
      if (d > 0) {
        s += ", ";
      }
      s += std::to_string(t.size(d));
    }
    s += "]";
    return s;
  }
  if (value.isList()) {
    auto list = value.toListRef();
    std::string s = "generic_list[" + std::to_string(list.size()) + "](";
    for (size_t i = 0; i < list.size(); ++i) {
      if (i > 0) {
        s += ", ";
      }
      s += traceIValue(list[i]);
    }
    s += ")";
    return s;
  }
  return value.tagKind() + ":" + ivalueToString(value);
}

void setGraphDevice(nativert::Graph* graph, bool isCuda) {
  c10::Device target = isCuda
      ? c10::Device(
            c10::kCUDA,
            facebook::velox::wave::currentDevice()
                ? static_cast<c10::DeviceIndex>(
                      facebook::velox::wave::currentDevice()->deviceId)
                : static_cast<c10::DeviceIndex>(0))
      : c10::Device(c10::kCPU);
  nativert::Placement placement(target);
  graph->applyDevicePlacement(placement);
}

namespace {

void printElement(std::ostream& os, const at::Tensor& flat, int64_t idx) {
  switch (flat.scalar_type()) {
    case c10::ScalarType::Float:
      os << flat[idx].item<float>();
      break;
    case c10::ScalarType::Double:
      os << flat[idx].item<double>();
      break;
    case c10::ScalarType::Long:
      os << flat[idx].item<int64_t>();
      break;
    case c10::ScalarType::Int:
      os << flat[idx].item<int32_t>();
      break;
    case c10::ScalarType::Short:
      os << flat[idx].item<int16_t>();
      break;
    case c10::ScalarType::Byte:
      os << static_cast<int>(flat[idx].item<uint8_t>());
      break;
    case c10::ScalarType::Half:
      os << flat[idx].item<at::Half>();
      break;
    case c10::ScalarType::BFloat16:
      os << flat[idx].item<at::BFloat16>();
      break;
    case c10::ScalarType::Bool:
      os << (flat[idx].item<bool>() ? "true" : "false");
      break;
    default:
      os << flat[idx].item<float>();
      break;
  }
}

void printTensorImpl(
    std::ostream& os,
    const at::Tensor& tensor,
    int64_t dim,
    int64_t offset,
    int64_t flatStride,
    const at::Tensor& flat) {
  os << "[";
  auto size = tensor.size(dim);
  if (size == 0) {
    os << "]";
    return;
  }
  if (dim == tensor.dim() - 1) {
    for (int64_t i = 0; i < size; ++i) {
      if (i > 0) {
        os << ", ";
      }
      printElement(os, flat, offset + i);
    }
  } else {
    auto innerStride = flatStride / size;
    for (int64_t i = 0; i < size; ++i) {
      if (i > 0) {
        os << ",\n";
        for (int64_t d = 0; d <= dim; ++d) {
          os << " ";
        }
      }
      printTensorImpl(
          os, tensor, dim + 1, offset + i * innerStride, innerStride, flat);
    }
  }
  os << "]";
}

} // namespace

std::string tensorDebugString(const at::Tensor& t, int32_t maxElements) {
  auto flat = t.cpu().contiguous().flatten();
  auto limit = maxElements > 0 ? std::min<int64_t>(flat.numel(), maxElements)
                               : flat.numel();
  std::stringstream ss;
  ss << "shape=" << t.sizes() << " dtype=" << t.dtype() << " [";
  for (int64_t i = 0; i < limit; ++i) {
    if (i > 0) {
      ss << ", ";
    }
    printElement(ss, flat, i);
  }
  if (flat.numel() > limit) {
    ss << ", ... (" << flat.numel() << " total)";
  }
  ss << "]";
  return ss.str();
}

std::string firstDifference(
    const at::Tensor& actual,
    const at::Tensor& expected) {
  if (actual.sizes() != expected.sizes()) {
    std::stringstream ss;
    ss << "shape mismatch: actual=" << actual.sizes()
       << " expected=" << expected.sizes();
    return ss.str();
  }
  auto flatA = actual.cpu().contiguous().flatten();
  auto flatE = expected.cpu().contiguous().flatten();
  auto numel = flatA.numel();

  int64_t firstIdx = -1;
  int64_t lastIdx = -1;
  int64_t diffCount = 0;

  auto scan = [&]<typename T>(T* aPtr, T* ePtr, auto compare) {
    for (int64_t i = 0; i < numel; ++i) {
      if (compare(aPtr[i], ePtr[i])) {
        if (firstIdx < 0) {
          firstIdx = i;
        }
        lastIdx = i;
        ++diffCount;
      }
    }
  };

  auto floatCompare = [](double a, double e) {
    return std::abs(a - e) > (1e-5 + 1e-4 * std::abs(e));
  };
  auto intCompare = [](auto a, auto e) { return a != e; };

  switch (actual.scalar_type()) {
    case c10::ScalarType::Float:
      scan(
          flatA.data_ptr<float>(),
          flatE.data_ptr<float>(),
          [&](float a, float e) { return floatCompare(a, e); });
      break;
    case c10::ScalarType::Double:
      scan(flatA.data_ptr<double>(), flatE.data_ptr<double>(), floatCompare);
      break;
    case c10::ScalarType::Half:
      scan(
          flatA.data_ptr<at::Half>(),
          flatE.data_ptr<at::Half>(),
          [&](at::Half a, at::Half e) {
            return floatCompare(static_cast<double>(a), static_cast<double>(e));
          });
      break;
    case c10::ScalarType::BFloat16:
      scan(
          flatA.data_ptr<at::BFloat16>(),
          flatE.data_ptr<at::BFloat16>(),
          [&](at::BFloat16 a, at::BFloat16 e) {
            return floatCompare(static_cast<double>(a), static_cast<double>(e));
          });
      break;
    case c10::ScalarType::Bool:
      scan(flatA.data_ptr<bool>(), flatE.data_ptr<bool>(), intCompare);
      break;
    case c10::ScalarType::Byte:
      scan(flatA.data_ptr<uint8_t>(), flatE.data_ptr<uint8_t>(), intCompare);
      break;
    case c10::ScalarType::Short:
      scan(flatA.data_ptr<int16_t>(), flatE.data_ptr<int16_t>(), intCompare);
      break;
    case c10::ScalarType::Int:
      scan(flatA.data_ptr<int32_t>(), flatE.data_ptr<int32_t>(), intCompare);
      break;
    default:
      scan(flatA.data_ptr<int64_t>(), flatE.data_ptr<int64_t>(), intCompare);
      break;
  }

  if (firstIdx < 0) {
    return "";
  }
  std::stringstream ss;
  ss << "first diff at [" << firstIdx << "/" << numel << "]: actual=";
  switch (actual.scalar_type()) {
    case c10::ScalarType::Float:
      ss << flatA.data_ptr<float>()[firstIdx]
         << " expected=" << flatE.data_ptr<float>()[firstIdx];
      break;
    case c10::ScalarType::Double:
      ss << flatA.data_ptr<double>()[firstIdx]
         << " expected=" << flatE.data_ptr<double>()[firstIdx];
      break;
    case c10::ScalarType::Half:
      ss << static_cast<float>(flatA.data_ptr<at::Half>()[firstIdx])
         << " expected="
         << static_cast<float>(flatE.data_ptr<at::Half>()[firstIdx]);
      break;
    case c10::ScalarType::BFloat16:
      ss << static_cast<float>(flatA.data_ptr<at::BFloat16>()[firstIdx])
         << " expected="
         << static_cast<float>(flatE.data_ptr<at::BFloat16>()[firstIdx]);
      break;
    case c10::ScalarType::Bool:
      ss << flatA.data_ptr<bool>()[firstIdx]
         << " expected=" << flatE.data_ptr<bool>()[firstIdx];
      break;
    case c10::ScalarType::Byte:
      ss << static_cast<int>(flatA.data_ptr<uint8_t>()[firstIdx])
         << " expected="
         << static_cast<int>(flatE.data_ptr<uint8_t>()[firstIdx]);
      break;
    case c10::ScalarType::Short:
      ss << flatA.data_ptr<int16_t>()[firstIdx]
         << " expected=" << flatE.data_ptr<int16_t>()[firstIdx];
      break;
    case c10::ScalarType::Int:
      ss << flatA.data_ptr<int32_t>()[firstIdx]
         << " expected=" << flatE.data_ptr<int32_t>()[firstIdx];
      break;
    default:
      ss << flatA.data_ptr<int64_t>()[firstIdx]
         << " expected=" << flatE.data_ptr<int64_t>()[firstIdx];
      break;
  }
  ss << ", " << diffCount << " diffs, last at [" << lastIdx << "]";
  return ss.str();
}

bool tensorsMatch(const at::Tensor& actual, const at::Tensor& expected) {
  if (actual.scalar_type() != expected.scalar_type() ||
      actual.sizes() != expected.sizes()) {
    return false;
  }
  if (at::isFloatingType(actual.scalar_type())) {
    return at::allclose(
        actual.cpu(), expected.cpu(), /*rtol=*/1e-4, /*atol=*/1e-5);
  }
  return actual.cpu().equal(expected.cpu());
}

std::string tensorToString(const at::Tensor& t) {
  auto cpu = t.cpu().contiguous();
  auto flat = cpu.flatten();
  std::stringstream ss;
  if (cpu.dim() == 0) {
    printElement(ss, flat, 0);
  } else {
    printTensorImpl(ss, cpu, 0, 0, flat.numel(), flat);
  }
  return ss.str();
}

void saveReferenceFrame(
    const nativert::ExecutionFrame& frame,
    const nativert::Graph& graph,
    const std::string& path) {
  c10::impl::GenericDict dict(c10::IntType::get(), c10::AnyType::get());
  for (const auto* value : graph.values()) {
    if (!value) {
      continue;
    }
    auto id = value->id();
    const auto& iv = frame.getIValue(id);
    if (iv.isNone()) {
      continue;
    }
    if (iv.isTensor()) {
      const auto& t = iv.toTensor();
      if (t.numel() > 0) {
        dict.insert(static_cast<int64_t>(id), c10::IValue(t.cpu()));
      }
    } else if (iv.isInt() || iv.isDouble() || iv.isBool()) {
      dict.insert(static_cast<int64_t>(id), iv);
    }
  }
  auto data = torch::jit::pickle_save(c10::IValue(dict));
  std::ofstream out(path, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void saveReferenceFrame(
    const nativert::ExecutionFrame& frame,
    int32_t numValues,
    const std::string& path) {
  c10::impl::GenericDict dict(c10::IntType::get(), c10::AnyType::get());
  for (int32_t id = 0; id < numValues; ++id) {
    const auto& iv = frame.getIValue(id);
    if (iv.isNone()) {
      continue;
    }
    if (iv.isTensor()) {
      const auto& t = iv.toTensor();
      if (t.numel() > 0) {
        dict.insert(static_cast<int64_t>(id), c10::IValue(t.cpu()));
      }
    } else if (iv.isInt() || iv.isDouble() || iv.isBool()) {
      dict.insert(static_cast<int64_t>(id), iv);
    }
  }
  auto data = torch::jit::pickle_save(c10::IValue(dict));
  std::ofstream out(path, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::unordered_map<int32_t, c10::IValue> loadReferenceFrame(
    const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(in.good(), "Cannot open reference frame: ", path);
  auto size = in.tellg();
  in.seekg(0);
  std::vector<char> data(size);
  in.read(data.data(), size);
  auto ivalue = torch::jit::pickle_load(data);
  TORCH_CHECK(ivalue.isGenericDict(), "Expected dict in reference frame");
  std::unordered_map<int32_t, c10::IValue> result;
  for (const auto& entry : ivalue.toGenericDict()) {
    result[static_cast<int32_t>(entry.key().toInt())] = entry.value();
  }
  return result;
}

std::string frameString(
    const nativert::ExecutionFrame& frame,
    int32_t valueId) {
  const auto& iv = frame.getIValue(valueId);
  return traceIValue(iv);
}

std::string refFrameString(int32_t valueId) {
  auto* ref = WaveConfig::get().referenceFrame;
  if (!ref) {
    return "none";
  }
  auto it = ref->find(valueId);
  if (it == ref->end()) {
    return "none";
  }
  return traceIValue(it->second);
}

std::string standaloneToString(NodeCP node) {
  auto boundaryInputs = inputValueSet(node);
  PrintOptions opts = NodePrinter::defaults();
  opts.boundaryValues = &boundaryInputs;
  opts.showOutputIds = true;
  opts.inlineIntermediates = false;
  return NodePrinter(opts).print(node);
}

bool isInPlaceMutation(NodeCP node, ValueCP value) {
  auto* schema = findFunctionSchema(node->target());
  if (schema) {
    const auto& inputs = node->inputs();
    const auto& args = schema->arguments();
    for (size_t i = 0; i < inputs.size() && i < args.size(); ++i) {
      if (inputs[i].value == value) {
        auto* aliasInfo = args[i].alias_info();
        return aliasInfo && aliasInfo->isWrite();
      }
    }
    return false;
  }
  auto target = std::string(node->target());
  return target.find("_.") != std::string::npos ||
      (target.size() > 1 && target.back() == '_');
}

TraceState parseTraceValues(const std::string& csv) {
  TraceState state;
  if (csv.empty()) {
    return state;
  }
  std::istringstream stream(csv);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      state.valueIds.insert(std::stoi(token));
    }
  }
  return state;
}

void traceFrameValues(
    const std::string& label,
    const std::vector<nativert::ValueId>& valueIds,
    nativert::ExecutionFrame& frame,
    TraceState& traceState) {
  if (traceState.empty()) {
    return;
  }
  auto maxElements = WaveConfig::get().tensorPrintElementLimit;
  for (auto id : valueIds) {
    if (!traceState.shouldTrace(id)) {
      continue;
    }
    const auto& iv = frame.getIValue(id);
    if (iv.isNone()) {
      continue;
    }
    traceState.markTraced(id);
    if (iv.isTensor()) {
      std::cout << "  trace " << label << " %" << id << ": "
                << tensorDebugString(iv.toTensor(), maxElements) << std::endl;
    } else {
      std::cout << "  trace " << label << " %" << id << ": " << traceIValue(iv)
                << std::endl;
    }
  }
}

} // namespace torch::wave
