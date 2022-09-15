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

#include <optional>
#include "velox/substrait/VeloxSubstraitSignature.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"
#include "velox/substrait/proto/substrait/plan.pb.h"
#include "velox/type/Type.h"

namespace facebook::velox::substrait {

struct ExtensionFunctionId {
  /// Substrait extension YAML file uri.
  std::string uri;

  /// Substrait signature used in the function extension declaration is a
  /// combination of the name of the function along with a list of input
  /// argument types.The format is as follows : <function
  /// name>:<short_arg_type0>_<short_arg_type1>_..._<short_arg_typeN> for more
  /// detail information about the argument type please refer to link
  /// https://substrait.io/extensions/#function-signature-compound-names
  std::string signature;

  bool operator==(const ExtensionFunctionId& other) const {
    return (uri == other.uri && signature == other.signature);
  }

  /// Given a function name and function arguments and return  a mating
  /// extension functionId.
  static ExtensionFunctionId create(
      const std::string& functionName,
      const std::vector<TypePtr>& arguments);
};

/// Assigns unique IDs to function signatures using ExtensionFunctionId.
class SubstraitExtensionCollector {
 public:
  SubstraitExtensionCollector();

  /// Given function name and arguments and return functionId using FunctionId.
  int getReferenceNumber(
      const std::string& functionName,
      const std::vector<TypePtr>& arguments);

  /// add extension functions to Substrait plan.
  void addExtensionsToPlan(::substrait::Plan* plan) const;

 private:
  /// A bi-direction hash map to keep the relation between reference number and
  /// either function or type signature.
  /// @T FunctionId
  template <class T>
  class BiDirectionHashMap {
   public:
    /// If the forwardMap_ does not contain the key, then the key and value will be written, otherwise the value will be overwritten,
    /// If the reverseMap_ does not contain the value, then the value and key will be written, otherwise the key will be overwritten.
    void put(const int& key, const T& value);

    std::map<int, T> forwardMap_;
    std::unordered_map<T, int> reverseMap_;
  };

  /// the count of extension function reference in a substrait plan.
  int functionReference_ = -1;
  /// extension function collected in substrait plan.
  std::shared_ptr<BiDirectionHashMap<ExtensionFunctionId>> extensionFunctions_;
};

using SubstraitExtensionCollectorPtr =
    std::shared_ptr<SubstraitExtensionCollector>;

} // namespace facebook::velox::substrait

namespace std {

/// hash function of facebook::velox::substrait::ExtensionFunctionId
template <>
struct hash<facebook::velox::substrait::ExtensionFunctionId> {
  size_t operator()(
      const facebook::velox::substrait::ExtensionFunctionId& k) const {
    size_t val = hash<std::string>()(k.uri);
    val = val * 31 + hash<std::string>()(k.signature);
    return val;
  }
};

}; // namespace std
