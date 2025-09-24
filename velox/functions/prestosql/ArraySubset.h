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

#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Udf.h"

namespace facebook::velox::functions {

/// Fast path for constant indices: array_subset(arr, array[1, 2, 3]).
template <typename TExec, typename T>
struct ArraySubsetFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Array<T>>* /*inputArray*/,
      const arg_type<Array<int32_t>>* indices) {
    if (indices != nullptr) {
      constantIndices_ = true;
      initializeIndices(*indices);
    }
  }

  void call(
      out_type<Array<T>>& out,
      const arg_type<Array<T>>& inputArray,
      const arg_type<Array<int32_t>>& indices) {
    // Defensive check and early return for empty indices
    if (indices.empty()) {
      return;
    }

    if (!constantIndices_) {
      validIndices_.clear();
      initializeIndices(indices);
    }

    // If no valid indices, return empty array
    if (validIndices_.empty()) {
      return;
    }

    // Sort indices to maintain order in output
    std::vector<int32_t> sortedIndices(
        validIndices_.begin(), validIndices_.end());
    std::sort(sortedIndices.begin(), sortedIndices.end());

    for (int32_t index : sortedIndices) {
      // Convert 1-based to 0-based index
      int32_t zeroBasedIndex = index - 1;

      // Check bounds - only add elements that exist in the input array
      if (zeroBasedIndex >= 0 && zeroBasedIndex < inputArray.size()) {
        if (inputArray[zeroBasedIndex].has_value()) {
          if constexpr (std::is_same_v<T, Varchar>) {
            out.add_item().setNoCopy(inputArray[zeroBasedIndex].value());
          } else {
            out.push_back(inputArray[zeroBasedIndex].value());
          }
        } else {
          // Include null elements in output
          out.add_null();
        }
      }
      // Note: Out-of-bounds indices are silently ignored (no element added)
    }
  }

 private:
  void initializeIndices(const arg_type<Array<int32_t>>& indices) {
    for (const auto& index : indices.skipNulls()) {
      // Only store positive indices (1-based)
      if (index > 0) {
        validIndices_.emplace(index);
      }
    }
  }

  bool constantIndices_{false};
  folly::F14FastSet<int32_t> validIndices_;
};

/// String version that avoids copy of strings.
template <typename TExec>
struct ArraySubsetVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  static constexpr int32_t reuse_strings_from_arg = 0;

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Array<Varchar>>* /*inputArray*/,
      const arg_type<Array<int32_t>>* indices) {
    if (indices != nullptr) {
      constantIndices_ = true;
      initializeIndices(*indices);
    }
  }

  void call(
      out_type<Array<Varchar>>& out,
      const arg_type<Array<Varchar>>& inputArray,
      const arg_type<Array<int32_t>>& indices) {
    // If indices is empty, return empty array
    if (indices.empty()) {
      return;
    }

    if (!constantIndices_) {
      validIndices_.clear();
      initializeIndices(indices);
    }

    // If no valid indices, return empty array
    if (validIndices_.empty()) {
      return;
    }

    // Sort indices to maintain order in output
    std::vector<int32_t> sortedIndices(
        validIndices_.begin(), validIndices_.end());
    std::sort(sortedIndices.begin(), sortedIndices.end());

    for (int32_t index : sortedIndices) {
      // Convert 1-based to 0-based index
      int32_t zeroBasedIndex = index - 1;

      // Check bounds - only add elements that exist in the input array
      if (zeroBasedIndex >= 0 && zeroBasedIndex < inputArray.size()) {
        if (inputArray[zeroBasedIndex].has_value()) {
          out.add_item().setNoCopy(inputArray[zeroBasedIndex].value());
        } else {
          // Include null elements in output
          out.add_null();
        }
      }
      // Note: Out-of-bounds indices are silently ignored (no element added)
    }
  }

 private:
  void initializeIndices(const arg_type<Array<int32_t>>& indices) {
    for (const auto& index : indices.skipNulls()) {
      // Only store positive indices (1-based)
      if (index > 0) {
        validIndices_.emplace(index);
      }
    }
  }

  bool constantIndices_{false};
  folly::F14FastSet<int32_t> validIndices_;
};

/// Generic implementation for complex types.
template <typename TExec>
struct ArraySubsetGenericFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Array<Generic<T1>>>& out,
      const arg_type<Array<Generic<T1>>>& inputArray,
      const arg_type<Array<int32_t>>& indices) {
    // If indices is empty, return empty array
    if (indices.empty()) {
      return;
    }

    // Collect valid indices with explicit bounds checking
    folly::F14FastSet<int32_t> validIndices;

    // Add bounds check to prevent integer overflow
    if (indices.size() > 0 &&
        indices.size() < std::numeric_limits<int32_t>::max()) {
      for (const auto& index : indices.skipNulls()) {
        // Only store positive indices (1-based) and check bounds
        if (index > 0 && index <= inputArray.size()) {
          validIndices.emplace(index);
        }
      }
    }

    // If no valid indices, return empty array
    if (validIndices.empty()) {
      return;
    }

    // Sort indices to maintain order in output
    std::vector<int32_t> sortedIndices(
        validIndices.begin(), validIndices.end());
    std::sort(sortedIndices.begin(), sortedIndices.end());

    for (int32_t index : sortedIndices) {
      // Convert 1-based to 0-based index
      int32_t zeroBasedIndex = index - 1;

      // Check bounds - only add elements that exist in the input array
      if (zeroBasedIndex >= 0 && zeroBasedIndex < inputArray.size()) {
        if (inputArray[zeroBasedIndex].has_value()) {
          out.push_back(inputArray[zeroBasedIndex].value());
        } else {
          // Include null elements in output
          out.add_null();
        }
      }
      // Note: Out-of-bounds indices are silently ignored (no element added)
    }
  }
};

} // namespace facebook::velox::functions
