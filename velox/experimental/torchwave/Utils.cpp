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

#include <type_traits>

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/StringUtil.h>
#include <fmt/format.h>

namespace torch::wave {

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

} // namespace torch::wave
