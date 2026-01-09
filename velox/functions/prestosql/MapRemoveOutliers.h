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

#include <algorithm>
#include <cmath>
#include <vector>
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {

template <typename TExec, typename Key, typename Value>
struct MapRemoveOutliersFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Key, Value>>& out,
      const arg_type<Map<Key, Value>>& inputMap,
      const arg_type<double>& threshold) {
    if (inputMap.empty()) {
      return;
    }

    std::vector<Value> values;
    values.reserve(inputMap.size());

    for (const auto& entry : inputMap) {
      if (entry.second.has_value()) {
        values.push_back(entry.second.value());
      }
    }

    if (values.empty()) {
      for (const auto& entry : inputMap) {
        if (!entry.second.has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter = entry.first;
        }
      }
      return;
    }

    double mean = 0.0;
    for (const auto& val : values) {
      mean += static_cast<double>(val);
    }
    mean /= values.size();

    double variance = 0.0;
    for (const auto& val : values) {
      double diff = static_cast<double>(val) - mean;
      variance += diff * diff;
    }
    variance /= values.size();
    double stddev = std::sqrt(variance);

    double lowerBound = mean - threshold * stddev;
    double upperBound = mean + threshold * stddev;

    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter = entry.first;
      } else {
        double val = static_cast<double>(entry.second.value());
        if (val >= lowerBound && val <= upperBound) {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter = entry.first;
          valueWriter = entry.second.value();
        }
      }
    }
  }
};

template <typename TExec, typename Key>
struct MapRemoveOutliersVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Key, Varchar>>& out,
      const arg_type<Map<Key, Varchar>>& inputMap,
      const arg_type<double>& /*threshold*/) {
    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter = entry.first;
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter = entry.first;
        valueWriter.copy_from(entry.second.value());
      }
    }
  }
};

template <typename TExec>
struct MapRemoveOutliersGenericFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Generic<T1>, Generic<T2>>>& out,
      const arg_type<Map<Generic<T1>, Generic<T2>>>& inputMap,
      const arg_type<double>& /*threshold*/) {
    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter.copy_from(entry.first);
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter.copy_from(entry.first);
        valueWriter.copy_from(entry.second.value());
      }
    }
  }
};

void registerMapRemoveOutliers(const std::string& name);

} // namespace facebook::velox::functions
