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

#include "velox/functions/Udf.h"
#include "velox/functions/prestosql/CheckedArithmetic.h"
#include "velox/type/Conversions.h"

namespace facebook::velox::functions {

template <typename TExecCtx, bool isMax>
struct ArrayMinMaxFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExecCtx);

  template <typename T>
  void update(T& currentValue, const T& candidateValue) {
    if constexpr (isMax) {
      if (candidateValue > currentValue) {
        currentValue = candidateValue;
      }
    } else {
      if (candidateValue < currentValue) {
        currentValue = candidateValue;
      }
    }
  }

  template <typename TReturn, typename TInput>
  void assign(TReturn& out, const TInput& value) {
    out = value;
  }

  void assign(out_type<Varchar>& out, const arg_type<Varchar>& value) {
    // TODO: reuse strings once support landed.
    out.resize(value.size());
    if (value.size() != 0) {
      std::memcpy(out.data(), value.data(), value.size());
    }
  }

  template <typename TReturn, typename TInput>
  FOLLY_ALWAYS_INLINE bool call(TReturn& out, const TInput& array) {
    // Result is null if array is empty.
    if (array.size() == 0) {
      return false;
    }

    if (!array.mayHaveNulls()) {
      // Input array does not have nulls.
      auto currentValue = *array[0];
      for (auto i = 1; i < array.size(); i++) {
        update(currentValue, array[i].value());
      }
      assign(out, currentValue);
      return true;
    }

    auto it = array.begin();
    // Result is null if any element is null.
    if (!it->has_value()) {
      return false;
    }

    auto currentValue = it->value();
    ++it;
    while (it != array.end()) {
      if (!it->has_value()) {
        return false;
      }
      update(currentValue, it->value());
      ++it;
    }

    assign(out, currentValue);
    return true;
  }
};

template <typename TExecCtx>
struct ArrayMinFunction : public ArrayMinMaxFunction<TExecCtx, false> {};

template <typename TExecCtx>
struct ArrayMaxFunction : public ArrayMinMaxFunction<TExecCtx, true> {};

template <typename TExecCtx, typename T>
struct ArrayJoinFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExecCtx);

  template <typename C>
  void writeValue(out_type<velox::Varchar>& result, const C& value) {
    bool nullOutput = false;
    result += util::Converter<TypeKind::VARCHAR, void, false>::cast(
        value, nullOutput);
  }

  template <typename C>
  void writeOutput(
      out_type<velox::Varchar>& result,
      const arg_type<velox::Varchar>& delim,
      const C& value,
      bool& firstNonNull) {
    if (!firstNonNull) {
      writeValue(result, delim);
    }
    writeValue(result, value);
    firstNonNull = false;
  }

  void createOutputString(
      out_type<velox::Varchar>& result,
      const arg_type<velox::Array<T>>& inputArray,
      const arg_type<velox::Varchar>& delim,
      std::optional<std::string> nullReplacement = std::nullopt) {
    bool firstNonNull = true;
    if (inputArray.size() == 0) {
      return;
    }

    for (const auto& entry : inputArray) {
      if (entry.has_value()) {
        writeOutput(result, delim, entry.value(), firstNonNull);
      } else if (nullReplacement.has_value()) {
        writeOutput(result, delim, nullReplacement.value(), firstNonNull);
      }
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<velox::Varchar>& result,
      const arg_type<velox::Array<T>>& inputArray,
      const arg_type<velox::Varchar>& delim) {
    createOutputString(result, inputArray, delim);
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<velox::Varchar>& result,
      const arg_type<velox::Array<T>>& inputArray,
      const arg_type<velox::Varchar>& delim,
      const arg_type<velox::Varchar>& nullReplacement) {
    createOutputString(result, inputArray, delim, nullReplacement.getString());
    return true;
  }
};

/// Function Signature: combinations(array(T), n) -> array(array(T))
/// Returns n-element combinations of the input array. If the input array has no
/// duplicates, combinations returns n-element subsets. Order of subgroup is
/// deterministic but unspecified. Order of elements within a subgroup are
/// deterministic but unspecified. n must not be greater than 5, and the total
/// size of subgroups generated must be smaller than 100000.
template <typename TExecParams, typename T>
struct CombinationsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  static constexpr int32_t kMaxCombinationSize = 5;
  static constexpr int64_t kMaxNumberOfCombinations = 100000;
  /// TODO: Add ability to re-use strings once reuse_strings_from_arg supports
  /// reusing strings nested within complex types.

  int64_t calculateCombinationCount(
      int64_t inputArraySize,
      int64_t combinationSize) {
    int64_t combinationCount = 1;
    for (int i = 1;
         i <= combinationSize && combinationCount <= kMaxNumberOfCombinations;
         i++, inputArraySize--) {
      combinationCount = (combinationCount * inputArraySize) / i;
    }
    return combinationCount;
  }

  void resetCombination(std::vector<int>& combination, int to) {
    std::iota(combination.begin(), combination.begin() + to, 0);
  }

  std::vector<int> firstCombination(int64_t size) {
    std::vector<int> combination(size, 0);
    std::iota(combination.begin(), combination.end(), 0);
    return combination;
  }

  bool nextCombination(std::vector<int>& combination, int64_t inputArraySize) {
    for (int i = 0; i < combination.size() - 1; i++) {
      if (combination[i] + 1 < combination[i + 1]) {
        combination[i]++;
        resetCombination(combination, i);
        return true;
      }
    }
    if (combination.size() > 0 && combination.back() + 1 < inputArraySize) {
      combination.back()++;
      resetCombination(combination, combination.size() - 1);
      return true;
    }
    return false;
  }

  void appendEntryFromCombination(
      out_type<velox::Array<velox::Array<T>>>& result,
      const arg_type<velox::Array<T>>& array,
      std::vector<int>& combination) {
    auto& innerArray = result.add_item();
    for (auto idx : combination) {
      if (!array[idx].has_value()) {
        innerArray.add_null();
        continue;
      }
      if constexpr (std::is_same_v<T, Varchar>) {
        innerArray.add_item().setNoCopy(array[idx].value());
      } else {
        innerArray.add_item() = array[idx].value();
      }
    }
  }

  /// Employs an iterative approach of generating combinations. Each
  /// 'combination' is a set of numbers that represent the indices of the
  /// elements within the input array. Later, using this, an entry is generated
  /// in the output array by copying the elements corresponding to those indices
  /// from the input array.
  /// The iterative approach is based on the fact that if combinations are
  /// lexicographically sorted based on their indices in the input array then
  /// each combination in that sequence is the next lowest lexicographic order
  /// that can be formed when compared to its previous consecutive combination.
  // So we start with the first combination (first k elements 0,1,2,...k the
  // lowest lexicographic order) and on each iteration generate the next lowest
  // lexicographic order. eg. for (0,1,2,3) combinations of size 3 are
  // (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
  FOLLY_ALWAYS_INLINE void call(
      out_type<velox::Array<velox::Array<T>>>& result,
      const arg_type<velox::Array<T>>& array,
      const int32_t& combinationSize) {
    auto arraySize = array.size();
    VELOX_USER_CHECK_GE(
        combinationSize,
        0,
        "combination size must not be negative: {}",
        combinationSize);
    VELOX_USER_CHECK_LE(
        combinationSize,
        kMaxCombinationSize,
        "combination size must not exceed {}: {}",
        kMaxCombinationSize,
        combinationSize);
    if (combinationSize > arraySize) {
      // An empty array should be returned
      return;
    }
    if (combinationSize == 0) {
      // return a single empty array element within the array.
      result.add_item();
      return;
    }
    int64_t combinationCount =
        calculateCombinationCount(arraySize, combinationSize);
    VELOX_USER_CHECK_LE(
        combinationCount,
        kMaxNumberOfCombinations,
        "combinations exceed max size of {}",
        kMaxNumberOfCombinations);
    result.reserve(combinationCount);
    std::vector<int> currentCombination = firstCombination(combinationSize);
    do {
      appendEntryFromCombination(result, array, currentCombination);
    } while (nextCombination(currentCombination, arraySize));
  }
};

template <typename T>
struct ArraySumFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T)
  template <typename TOutput, typename TInput>
  FOLLY_ALWAYS_INLINE void call(TOutput& out, const TInput& array) {
    TOutput sum = 0;
    for (const auto& item : array) {
      if (item.has_value()) {
        if constexpr (std::is_same_v<TOutput, int64_t>) {
          sum = checkedPlus<TOutput>(sum, *item);
        } else {
          sum += *item;
        }
      }
    }
    out = sum;
    return;
  }

  template <typename TOutput, typename TInput>
  FOLLY_ALWAYS_INLINE void callNullFree(TOutput& out, const TInput& array) {
    // Not nulls path
    TOutput sum = 0;
    for (const auto& item : array) {
      if constexpr (std::is_same_v<TOutput, int64_t>) {
        sum = checkedPlus<TOutput>(sum, item);
      } else {
        sum += item;
      }
    }
    out = sum;
    return;
  }
};

template <typename T>
struct ArrayAverageFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      double& out,
      const arg_type<Array<double>>& array) {
    // If the array is empty, then set result to null.
    if (array.size() == 0) {
      return false;
    }

    double sum = 0;
    size_t count = 0;
    for (const auto& item : array) {
      if (item.has_value()) {
        sum += *item;
        count++;
      }
    }
    if (count != 0) {
      out = sum / count;
    }
    return count != 0;
  }

  FOLLY_ALWAYS_INLINE bool callNullFree(
      double& out,
      const null_free_arg_type<Array<double>>& array) {
    // If the array is empty, then set result to null.
    if (array.size() == 0) {
      return false;
    }

    double sum = 0;
    for (const auto& item : array) {
      sum += item;
    }
    out = sum / array.size();
    return true;
  }
};

template <typename TExecCtx, typename T>
struct ArrayHasDuplicatesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExecCtx);

  FOLLY_ALWAYS_INLINE void call(
      bool& out,
      const arg_type<velox::Array<T>>& inputArray) {
    folly::F14FastSet<arg_type<T>> uniqSet;
    int16_t numNulls = 0;
    out = false;
    for (const auto& item : inputArray) {
      if (item.has_value()) {
        if (!uniqSet.insert(item.value()).second) {
          out = true;
          break;
        }
      } else {
        numNulls++;
        if (numNulls == 2) {
          out = true;
          break;
        }
      }
    }
  }

  FOLLY_ALWAYS_INLINE void callNullFree(
      bool& out,
      const null_free_arg_type<velox::Array<T>>& inputArray) {
    folly::F14FastSet<null_free_arg_type<T>> uniqSet;
    out = false;
    for (const auto& item : inputArray) {
      if (!uniqSet.insert(item).second) {
        out = true;
        break;
      }
    }
  }
};

// Function Signature: array<T> -> map<T, int>, where T is ("bigint", "varchar")
// Returns a map with frequency of each element in the input array vector.
template <typename TExecParams, typename T>
struct ArrayFrequencyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void call(
      out_type<velox::Map<T, int>>& out,
      arg_type<velox::Array<T>> inputArray) {
    frequencyCount_.clear();

    for (const auto& item : inputArray.skipNulls()) {
      frequencyCount_[item]++;
    }

    // To make the output order of key value pairs deterministic (since F14
    // does not provide ordering guarantees), we do another iteration in the
    // input and look up element frequencies in the F14 map. To prevent
    // duplicates in the output, we remove the keys we already added.
    for (const auto& item : inputArray.skipNulls()) {
      auto it = frequencyCount_.find(item);
      if (it != frequencyCount_.end()) {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter = item;
        valueWriter = it->second;
        frequencyCount_.erase(it);
      }
    }
  }

 private:
  folly::F14FastMap<arg_type<T>, int> frequencyCount_;
};

template <typename TExecParams, typename T>
struct ArrayNormalizeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void callNullFree(
      out_type<velox::Array<T>>& result,
      const null_free_arg_type<velox::Array<T>>& inputArray,
      const null_free_arg_type<T>& p) {
    VELOX_USER_CHECK_GE(
        p, 0, "array_normalize only supports non-negative p: {}", p);

    // If the input array is empty, then the empty result should be returned,
    // same as Presto.
    if (inputArray.size() == 0) {
      return;
    }

    result.reserve(inputArray.size());

    // If p = 0, then it is L0 norm. Presto version returns the input array.
    if (p == 0) {
      result.add_items(inputArray);
      return;
    }

    // Calculate p-norm.
    T sum = 0;
    for (const auto& item : inputArray) {
      sum += pow(abs(item), p);
    }

    T pNorm = pow(sum, 1.0 / p);

    // If the input array is a zero vector then pNorm = 0.
    // Return the input array for this case, same as Presto.
    if (pNorm == 0) {
      result.add_items(inputArray);
      return;
    }

    // Construct result array from the input array and pNorm.
    for (const auto& item : inputArray) {
      result.add_item() = item / pNorm;
    }
  }
};

} // namespace facebook::velox::functions
