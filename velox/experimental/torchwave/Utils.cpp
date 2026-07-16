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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
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
    // For non-contiguous tensors also show the strides, since shape alone does
    // not capture the layout.
    if (t.defined() && !t.is_contiguous()) {
      s += " strides[";
      for (int64_t d = 0; d < t.dim(); ++d) {
        if (d > 0) {
          s += ", ";
        }
        s += std::to_string(t.stride(d));
      }
      s += "]";
    }
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

  if (!isCuda) {
    return;
  }
  // Factory ops (zeros, ones, full, empty, arange, ...) with a `device` schema
  // parameter but no device attribute default to CPU.  applyDevicePlacement
  // only rewrites existing device attributes, so their outputs would land on
  // CPU and trip device-mismatch checks when combined with GPU tensors.  Inject
  // device=target for any such node so the whole graph runs on the GPU.
  for (auto& node : graph->nodes()) {
    auto nodeTarget = node.target();
    std::vector<std::string_view> atoms = c10::split(nodeTarget, '.');
    if (atoms.size() < 3 || atoms[0] != "torch" || atoms[1] != "ops") {
      continue;
    }
    if (node.tryGetAttribute("device") != nullptr ||
        node.tryGetInput("device") != nullptr) {
      continue;
    }
    auto schema = c10::Dispatcher::singleton().findSchema(
        {fmt::format(
             "{}::{}", atoms[atoms.size() - 3], atoms[atoms.size() - 2]),
         std::string(atoms[atoms.size() - 1])});
    if (!schema) {
      continue;
    }
    bool hasDeviceArg = false;
    for (const auto& arg : schema->schema().arguments()) {
      if (arg.name() == "device") {
        hasDeviceArg = true;
        break;
      }
    }
    if (hasDeviceArg) {
      node.addAttribute(nativert::Attribute{"device", target});
    }
  }
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
  ss << "shape=" << t.sizes() << " dtype=" << t.dtype();
  // For non-contiguous tensors also show the strides, since shape alone does
  // not capture the layout.
  if (t.defined() && !t.is_contiguous()) {
    ss << " strides=" << t.strides();
  }
  ss << " [";
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

std::optional<at::Tensor> scalarLikeToTensor(const c10::IValue& iv) {
  auto longOpts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  auto doubleOpts = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
  if (iv.isInt()) {
    return at::tensor(std::vector<int64_t>{iv.toInt()}, longOpts);
  }
  if (iv.isDouble()) {
    return at::tensor(std::vector<double>{iv.toDouble()}, doubleOpts);
  }
  if (iv.isBool()) {
    return at::tensor(std::vector<int64_t>{iv.toBool() ? 1 : 0}, longOpts)
        .to(at::kBool);
  }
  if (iv.isIntList()) {
    const auto& l = iv.toIntList();
    return at::tensor(std::vector<int64_t>(l.begin(), l.end()), longOpts);
  }
  if (iv.isDoubleList()) {
    const auto& l = iv.toDoubleList();
    return at::tensor(std::vector<double>(l.begin(), l.end()), doubleOpts);
  }
  if (iv.isBoolList()) {
    const auto& l = iv.toBoolList();
    std::vector<int64_t> v;
    v.reserve(l.size());
    for (bool b : l) {
      v.push_back(b ? 1 : 0);
    }
    return at::tensor(v, longOpts).to(at::kBool);
  }
  return std::nullopt;
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

// Reference-frame binary format ("TWREF\0\0\1").
//
// The reference frame is a {valueId -> tensor|scalar} map that can be very
// large (10K+ tensors, multiple GB). The flat-pickle format (torch::jit::
// pickle_save) inlines every tensor's bytes in one stream, so loading it
// requires reading the whole file and then memcpy-ing every tensor into a
// freshly allocated storage, all single-threaded.
//
// This format stores each tensor's raw bytes as a separately addressable,
// 64-byte-aligned region after a small header. Loading mmaps the file and
// constructs each tensor as a zero-copy view (at::from_blob) over the mapping,
// in parallel. No per-tensor allocation or copy happens. The mmap is kept
// alive for the lifetime of the returned tensors via a shared_ptr captured in
// each tensor's deleter. Endianness is host-native (internal tool).
namespace {

constexpr char kRefMagic[8] = {'T', 'W', 'R', 'E', 'F', '\0', '\0', '1'};

enum RefKind : int32_t {
  kRefTensor = 0,
  kRefInt = 1,
  kRefDouble = 2,
  kRefBool = 3,
};

uint64_t roundUp64(uint64_t x) {
  return (x + 63) & ~static_cast<uint64_t>(63);
}

struct RefEntry {
  int32_t id{0};
  int32_t kind{0};
  int32_t scalarType{0}; // tensor only
  std::vector<int64_t> dims; // tensor only
  uint64_t offset{0}; // tensor: byte offset of blob; scalar: raw value bits
  uint64_t length{0}; // tensor: blob byte length; scalar: 0
};

// RAII holder for an mmap region, shared across all tensors that view it.
// Non-copyable so the mapping is unmapped exactly once, when the last tensor
// viewing it is destroyed.
struct MmapHolder {
  void* addr;
  size_t size;
  MmapHolder(void* a, size_t s) : addr(a), size(s) {}
  MmapHolder(const MmapHolder&) = delete;
  MmapHolder& operator=(const MmapHolder&) = delete;
  MmapHolder(MmapHolder&&) = delete;
  MmapHolder& operator=(MmapHolder&&) = delete;
  ~MmapHolder() {
    if (addr != nullptr && addr != MAP_FAILED) {
      ::munmap(addr, size);
    }
  }
};

template <typename T>
void appendPod(std::vector<char>& buf, const T& v) {
  const char* p = reinterpret_cast<const char*>(&v);
  buf.insert(buf.end(), p, p + sizeof(T));
}

// Serializes a list of (id, IValue) entries to the TWREF format.
void serializeReferenceFrame(
    const std::vector<std::pair<int32_t, c10::IValue>>& items,
    const std::string& path) {
  // Hold contiguous CPU copies of tensors alive until they are written.
  std::vector<at::Tensor> heldTensors;
  std::vector<RefEntry> entries;
  std::vector<const void*> blobs;
  entries.reserve(items.size());
  blobs.reserve(items.size());

  for (const auto& [id, iv] : items) {
    RefEntry e{};
    e.id = id;
    // Tensors are stored as-is; scalars and scalar lists are wrapped into a 1-D
    // tensor of the matching dtype so they live on the same (kRefTensor) path
    // and can be compared element-wise at check time.
    at::Tensor t;
    if (iv.isTensor()) {
      t = iv.toTensor();
    } else if (auto st = scalarLikeToTensor(iv)) {
      t = *st;
    } else {
      continue;
    }
    t = t.cpu().contiguous();
    heldTensors.push_back(t);
    e.kind = kRefTensor;
    e.scalarType = static_cast<int32_t>(t.scalar_type());
    e.dims = t.sizes().vec();
    e.length = static_cast<uint64_t>(t.nbytes());
    blobs.push_back(t.data_ptr());
    entries.push_back(std::move(e));
  }

  // Compute header size and per-tensor blob offsets.
  uint64_t headerSize = sizeof(kRefMagic) + sizeof(uint32_t);
  for (const auto& e : entries) {
    headerSize += 4 + 4 + 4 + 4; // id, kind, scalarType, ndim
    headerSize += sizeof(int64_t) * e.dims.size(); // dims
    headerSize += 8 + 8; // offset, length
  }
  uint64_t dataStart = roundUp64(headerSize);
  uint64_t cursor = dataStart;
  for (auto& e : entries) {
    if (e.kind == kRefTensor) {
      e.offset = cursor;
      cursor += roundUp64(e.length);
    }
  }

  // Build the header buffer.
  std::vector<char> header;
  header.reserve(headerSize);
  header.insert(header.end(), kRefMagic, kRefMagic + sizeof(kRefMagic));
  appendPod(header, static_cast<uint32_t>(entries.size()));
  for (const auto& e : entries) {
    appendPod(header, e.id);
    appendPod(header, e.kind);
    appendPod(header, e.scalarType);
    appendPod(header, static_cast<int32_t>(e.dims.size()));
    for (int64_t d : e.dims) {
      appendPod(header, d);
    }
    appendPod(header, e.offset);
    appendPod(header, e.length);
  }

  std::ofstream out(path, std::ios::binary);
  TORCH_CHECK(out.good(), "Cannot open reference frame for write: ", path);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  // Pad to dataStart.
  static const char zeros[64] = {0};
  uint64_t pad = dataStart - header.size();
  out.write(zeros, static_cast<std::streamsize>(pad));
  // Write each tensor blob, 64-byte padded.
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    if (e.kind != kRefTensor) {
      continue;
    }
    out.write(
        reinterpret_cast<const char*>(blobs[i]),
        static_cast<std::streamsize>(e.length));
    uint64_t bpad = roundUp64(e.length) - e.length;
    while (bpad > 0) {
      auto n = std::min<uint64_t>(bpad, sizeof(zeros));
      out.write(zeros, static_cast<std::streamsize>(n));
      bpad -= n;
    }
  }
}

// Fallback loader for the legacy flat-pickle format.
std::unordered_map<int32_t, c10::IValue> loadReferenceFramePickle(
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

} // namespace

void saveReferenceFrame(
    const nativert::ExecutionFrame& frame,
    const nativert::Graph& graph,
    const std::string& path) {
  std::vector<std::pair<int32_t, c10::IValue>> items;
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
      if (iv.toTensor().numel() > 0) {
        items.emplace_back(static_cast<int32_t>(id), iv);
      }
    } else if (
        iv.isInt() || iv.isDouble() || iv.isBool() || iv.isIntList() ||
        iv.isDoubleList() || iv.isBoolList()) {
      items.emplace_back(static_cast<int32_t>(id), iv);
    }
  }
  serializeReferenceFrame(items, path);
}

void saveReferenceFrame(
    const nativert::ExecutionFrame& frame,
    int32_t numValues,
    const std::string& path) {
  std::vector<std::pair<int32_t, c10::IValue>> items;
  for (int32_t id = 0; id < numValues; ++id) {
    const auto& iv = frame.getIValue(id);
    if (iv.isNone()) {
      continue;
    }
    if (iv.isTensor()) {
      if (iv.toTensor().numel() > 0) {
        items.emplace_back(id, iv);
      }
    } else if (
        iv.isInt() || iv.isDouble() || iv.isBool() || iv.isIntList() ||
        iv.isDoubleList() || iv.isBoolList()) {
      items.emplace_back(id, iv);
    }
  }
  serializeReferenceFrame(items, path);
}

void saveReferenceFrame(
    const nativert::ExecutionFrame& frame,
    const nativert::Graph& graph,
    const std::unordered_map<int64_t, at::Tensor>& capturedTensors,
    const std::string& path) {
  std::vector<std::pair<int32_t, c10::IValue>> items;
  for (const auto* value : graph.values()) {
    if (!value) {
      continue;
    }
    auto id = static_cast<int64_t>(value->id());
    // Prefer the copy captured the instant the node ran; it cannot have been
    // corrupted by a later in-place write.
    auto it = capturedTensors.find(id);
    if (it != capturedTensors.end()) {
      if (it->second.numel() > 0) {
        items.emplace_back(static_cast<int32_t>(id), c10::IValue(it->second));
      }
      continue;
    }
    const auto& iv = frame.getIValue(value->id());
    if (iv.isInt() || iv.isDouble() || iv.isBool() || iv.isIntList() ||
        iv.isDoubleList() || iv.isBoolList()) {
      items.emplace_back(static_cast<int32_t>(id), iv);
    }
  }
  serializeReferenceFrame(items, path);
}

void saveTensorList(
    const std::vector<at::Tensor>& tensors,
    const std::string& path) {
  c10::List<at::Tensor> list;
  for (const auto& t : tensors) {
    if (t.defined()) {
      list.push_back(t.cpu());
    }
  }
  auto data = torch::jit::pickle_save(c10::IValue(list));
  std::ofstream out(path, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void saveIValueList(
    const std::vector<c10::IValue>& values,
    const std::string& path) {
  // A tuple (not a typed list) so pickling emits no element type tag: a
  // heterogeneous List[Any] fails pickleLoadWithTypes, but a tuple unpickles
  // generically and loadReferenceValues flattens it back to these elements.
  std::vector<c10::IValue> elems;
  elems.reserve(values.size());
  for (const auto& v : values) {
    elems.push_back(v.isTensor() ? c10::IValue(v.toTensor().cpu()) : v);
  }
  auto data = torch::jit::pickle_save(
      c10::IValue(c10::ivalue::Tuple::create(std::move(elems))));
  std::ofstream out(path, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::vector<at::Tensor> loadTensorList(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.good()) {
    return {};
  }
  auto size = in.tellg();
  in.seekg(0);
  std::vector<char> data(size);
  in.read(data.data(), size);
  auto ivalue = torch::jit::pickle_load(data);
  TORCH_CHECK(ivalue.isList(), "Expected list in tensor list file");
  std::vector<at::Tensor> result;
  for (const auto& element : ivalue.toListRef()) {
    result.push_back(element.toTensor());
  }
  return result;
}

std::unordered_map<int32_t, c10::IValue> loadReferenceFrame(
    const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  TORCH_CHECK(fd >= 0, "Cannot open reference frame: ", path);
  struct stat st{};
  TORCH_CHECK(::fstat(fd, &st) == 0, "fstat failed: ", path);
  size_t fileSize = static_cast<size_t>(st.st_size);

  // Detect the format from the magic header; fall back to flat pickle.
  bool isTwref = false;
  if (fileSize >= sizeof(kRefMagic)) {
    char magic[sizeof(kRefMagic)];
    isTwref = (::pread(fd, magic, sizeof(magic), 0) ==
               static_cast<ssize_t>(sizeof(magic))) &&
        std::memcmp(magic, kRefMagic, sizeof(kRefMagic)) == 0;
  }
  if (!isTwref) {
    ::close(fd);
    return loadReferenceFramePickle(path);
  }

  void* addr = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  TORCH_CHECK(addr != MAP_FAILED, "mmap failed: ", path);
  ::madvise(addr, fileSize, MADV_WILLNEED);
  auto holder = std::make_shared<MmapHolder>(addr, fileSize);
  const char* base = static_cast<const char*>(addr);

  // Parse the (small) header sequentially.
  size_t cursor = sizeof(kRefMagic);
  uint32_t count = 0;
  std::memcpy(&count, base + cursor, sizeof(count));
  cursor += sizeof(count);

  std::vector<RefEntry> entries(count);
  for (uint32_t i = 0; i < count; ++i) {
    RefEntry& e = entries[i];
    std::memcpy(&e.id, base + cursor, 4);
    cursor += 4;
    std::memcpy(&e.kind, base + cursor, 4);
    cursor += 4;
    std::memcpy(&e.scalarType, base + cursor, 4);
    cursor += 4;
    int32_t ndim = 0;
    std::memcpy(&ndim, base + cursor, 4);
    cursor += 4;
    e.dims.resize(ndim);
    for (int32_t d = 0; d < ndim; ++d) {
      std::memcpy(&e.dims[d], base + cursor, sizeof(int64_t));
      cursor += sizeof(int64_t);
    }
    std::memcpy(&e.offset, base + cursor, 8);
    cursor += 8;
    std::memcpy(&e.length, base + cursor, 8);
    cursor += 8;
  }

  // Build the IValues in parallel: tensors are zero-copy views over the mmap.
  std::vector<c10::IValue> values(count);
  at::parallel_for(
      0, count, /*grain_size=*/64, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          const RefEntry& e = entries[i];
          switch (e.kind) {
            case kRefTensor: {
              auto options =
                  at::TensorOptions()
                      .dtype(static_cast<at::ScalarType>(e.scalarType))
                      .device(at::kCPU);
              void* data =
                  const_cast<void*>(static_cast<const void*>(base + e.offset));
              values[i] = c10::IValue(
                  at::from_blob(
                      data,
                      e.dims,
                      [holder](
                          void*) { /* mapping freed when last tensor dies */ },
                      options));
              break;
            }
            case kRefInt: {
              int64_t v = 0;
              std::memcpy(&v, &e.offset, sizeof(int64_t));
              values[i] = c10::IValue(v);
              break;
            }
            case kRefDouble: {
              double v = 0;
              std::memcpy(&v, &e.offset, sizeof(double));
              values[i] = c10::IValue(v);
              break;
            }
            case kRefBool: {
              values[i] = c10::IValue(e.offset != 0);
              break;
            }
            default:
              break;
          }
        }
      });

  std::unordered_map<int32_t, c10::IValue> result;
  result.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    result.emplace(entries[i].id, std::move(values[i]));
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

namespace {
// True if two alias annotations name a common alias set (i.e. the values share
// storage), e.g. self Tensor(a!) and return Tensor(a!) on add_.
bool aliasSetsIntersect(const c10::AliasInfo& a, const c10::AliasInfo& b) {
  for (const auto& sym : a.beforeSets()) {
    if (b.beforeSets().count(sym) != 0) {
      return true;
    }
  }
  return false;
}
} // namespace

ValueCP schemaAliasedInput(NodeCP node, ValueCP output) {
  // The c10 FunctionSchema tells whether an output aliases (shares storage
  // with) an input: a view annotates both as the same alias set 'a'
  // (Tensor(a) self -> Tensor(a)), an in-place op as 'a!' (Tensor(a!) self ->
  // Tensor(a!)). Return the aliased input value, or nullptr if the output is
  // fresh (its return has no alias set, or no input shares it).
  const auto* schema = findFunctionSchema(node->target());
  if (schema == nullptr) {
    return nullptr;
  }
  const auto& outputs = node->outputs();
  int32_t outIdx = -1;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i] == output) {
      outIdx = static_cast<int32_t>(i);
      break;
    }
  }
  if (outIdx < 0 || static_cast<size_t>(outIdx) >= schema->returns().size()) {
    return nullptr;
  }
  const auto* outAlias = schema->returns()[outIdx].alias_info();
  if (outAlias == nullptr) {
    return nullptr;
  }
  const auto& args = schema->arguments();
  const auto& inputs = node->inputs();
  for (size_t j = 0; j < inputs.size() && j < args.size(); ++j) {
    const auto* inAlias = args[j].alias_info();
    if (inAlias != nullptr && aliasSetsIntersect(*inAlias, *outAlias)) {
      return inputs[j].value;
    }
  }
  return nullptr;
}

ValueCP viewStorageBase(ValueCP value) {
  // Follow storage-aliasing edges from a value to its base. An output aliases
  // an input when they share a c10 alias set -- a view (Tensor(a) self ->
  // Tensor(a)) or an in-place op (Tensor(a!) self -> Tensor(a!)). Prefer the
  // c10 schema (authoritative), and fall back to the torchwave viewOfArg
  // metadata for tw.* ops that have no c10 alias annotations. Follow so that a,
  // a.view(), and a.add_(...) all resolve to the same base; bounded by graph
  // acyclicity.
  while (value != nullptr) {
    auto* producer = value->producer();
    if (producer == nullptr) {
      break;
    }
    ValueCP next = schemaAliasedInput(producer, value);

    if (next == nullptr) {
      const auto* meta = Registry::metadata(producer->target());
      if (meta != nullptr && meta->viewOfArg.has_value()) {
        auto ordinal = *meta->viewOfArg;
        if (ordinal >= 0 &&
            static_cast<size_t>(ordinal) < producer->inputs().size()) {
          next = producer->inputs()[ordinal].value;
        }
      }
    }

    if (next == nullptr || next == value) {
      break;
    }
    value = next;
  }
  return value;
}

std::vector<ValueCP> dataMutatedInputs(NodeCP node) {
  std::vector<ValueCP> result;
  auto target = node->target();
  // detach_/lift_fresh_ carry a write-alias annotation but do not change tensor
  // data (autograd bookkeeping only); ignore them for data dependencies.
  if (target.find("detach") != std::string_view::npos ||
      target.find("lift_fresh") != std::string_view::npos) {
    return result;
  }
  const auto* schema = findFunctionSchema(target);
  if (schema == nullptr) {
    // No schema: fall back to the trailing-underscore convention used by
    // isInPlaceMutation, assuming the first input is the mutated self.
    bool inPlace = target.find("_.") != std::string_view::npos ||
        (!target.empty() && target.back() == '_');
    if (inPlace && !node->inputs().empty() && node->inputs()[0].value) {
      result.push_back(node->inputs()[0].value);
    }
    return result;
  }
  const auto& inputs = node->inputs();
  const auto& args = schema->arguments();
  for (size_t i = 0; i < inputs.size() && i < args.size(); ++i) {
    const auto* aliasInfo = args[i].alias_info();
    if (aliasInfo != nullptr && aliasInfo->isWrite() && inputs[i].value) {
      result.push_back(inputs[i].value);
    }
  }
  return result;
}

bool baseMutatedAfter(
    const nativert::Graph& graph,
    NodeCP afterNode,
    ValueCP value) {
  auto* base = viewStorageBase(value);
  bool seenAfter = false;
  for (const auto& node : graph.nodes()) {
    if (!seenAfter) {
      if (&node == afterNode) {
        seenAfter = true;
      }
      continue;
    }
    for (auto* mutated : dataMutatedInputs(&node)) {
      if (viewStorageBase(mutated) == base) {
        return true;
      }
    }
  }
  return false;
}

bool isUnreadyNoneDependency(ValueCP value, nativert::ExecutionFrame& frame) {
  // A value whose static type is None (an `asNone` optional argument) is
  // legitimately None and will never be produced -- it must not block its
  // consumer.  Only a runtime-None value with a non-None static type is an
  // unready dependency awaiting a producer.
  return value->type().kind() != nativert::Type::Kind::None &&
      frame.getIValue(value->id()).isNone();
}

bool nodeOutputsComputed(NodeCP node, nativert::ExecutionFrame& frame) {
  const auto& outputs = node->outputs();
  if (outputs.empty()) {
    return false;
  }
  for (auto* output : outputs) {
    if (frame.getIValue(output->id()).isNone()) {
      return false;
    }
  }
  return true;
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
    traceState.markTraced(id);
    if (iv.isNone()) {
      // Print a marker rather than skipping: an unset/freed value is exactly
      // what we want to see when tracing (e.g. a user input read as null on a
      // second run of a reused frame).
      std::cout << "  trace " << label << " %" << id << ": <none/unset>"
                << std::endl;
    } else if (iv.isTensor()) {
      std::cout << "  trace " << label << " %" << id << ": "
                << tensorDebugString(iv.toTensor(), maxElements) << std::endl;
    } else {
      std::cout << "  trace " << label << " %" << id << ": " << traceIValue(iv)
                << std::endl;
    }
  }
}

} // namespace torch::wave
