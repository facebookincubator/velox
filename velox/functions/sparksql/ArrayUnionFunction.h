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

namespace facebook::velox::functions::sparksql {

/// This class implements the array union function.
///
/// DEFINITION:
/// array_union(x, y) → array
/// Returns an array of the elements in the union of x and y, without
/// duplicates.
template <typename T>
struct ArrayUnionFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T)

  // Fast path for primitives.
  template <typename Out, typename In>
  void call(Out& out, const In& inputArray1, const In& inputArray2) {
    folly::F14FastSet<typename In::element_t> elementSet;
    bool nullAdded = false;
    bool nanAdded = false;
    auto addItems = [&](auto& inputArray) {
      for (const auto& item : inputArray) {
        if (item.has_value()) {
          if constexpr (
              std::is_same_v<In, arg_type<Array<float>>> ||
              std::is_same_v<In, arg_type<Array<double>>>) {
            bool isNaN = std::isnan(item.value());
            if ((isNaN && !nanAdded) ||
                (!isNaN && elementSet.insert(item.value()).second)) {
              auto& newItem = out.add_item();
              newItem = item.value();
            }
            if (!nanAdded && isNaN) {
              nanAdded = true;
            }
          } else if (elementSet.insert(item.value()).second) {
            auto& newItem = out.add_item();
            newItem = item.value();
          }
        } else if (!nullAdded) {
          nullAdded = true;
          out.add_null();
        }
      }
    };
    addItems(inputArray1);
    addItems(inputArray2);
  }

  void call(
      out_type<Array<Generic<T1>>>& out,
      const arg_type<Array<Generic<T1>>>& inputArray1,
      const arg_type<Array<Generic<T1>>>& inputArray2) {
    folly::F14FastSet<exec::GenericView> elementSet;
    bool nullAdded = false;
    auto addItems = [&](auto& inputArray) {
      for (const auto& item : inputArray) {
        if (item.has_value()) {
          if (elementSet.insert(item.value()).second) {
            auto& newItem = out.add_item();
            newItem.copy_from(item.value());
          }
        } else if (!nullAdded) {
          nullAdded = true;
          out.add_null();
        }
      }
    };
    addItems(inputArray1);
    addItems(inputArray2);
  }
};
} // namespace facebook::velox::functions::sparksql
