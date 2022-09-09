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
#include "velox/substrait/proto/substrait/algebra.pb.h"
#include "velox/substrait/proto/substrait/plan.pb.h"
#include "velox/type/Type.h"

namespace facebook::velox::substrait {

struct FunctionId {
  /// function uri
  std::string uri;
  /// function signature
  std::string signature;

  bool operator==(const FunctionId& other) const {
    return (uri == other.uri && signature == other.signature);
  }
};

struct TypeId {
  /// type uri
  std::string uri;
  /// type name
  std::string name;

  bool operator==(const TypeId& other) const {
    return (uri == other.uri && name == other.name);
  }
};

/// A ExtensionIdResolver is intend to used to resolve functionId or typeId
class ExtensionIdResolver {
 public:
  /// resolve functionId by given velox function name and function
  /// arguments .
  FunctionId resolve(
      const std::string& functionName,
      const std::vector<TypePtr>& arguments) const;

  /// resolve typeId by given velox type
  TypeId resolve(const TypePtr& type) const;
};

using ExtensionIdResolverPtr = std::shared_ptr<const ExtensionIdResolver>;

/// Maintains a mapping for function and function reference
class SubstraitExtensionCollector {
 public:
  SubstraitExtensionCollector();
  /// get function reference by function name and arguments.
  std::optional<int> getFunctionReference(
      const std::string& functionName,
      const std::vector<TypePtr>& arguments);

  /// get function reference by function name and arguments.
  std::optional<int> getTypeReference(const TypePtr& type);

  /// add extension functions and types to Substrait plan.
  void addExtensionsToPlan(::substrait::Plan* plan) const;

 private:
  /// A bi-direction hash map to keep the relation between reference number and
  /// either function or type anchor.
  /// @T either FunctionAnchor or TypeAnchor
  template <class T>
  class BiDirectionHashMap {
   public:
    void put(const int& key, const T& value);
    std::unordered_map<int, T> forwardMap_;
    std::unordered_map<T, int> reverseMap_;
  };

  /// add extension functions to Substrait plan.
  void addExtensionFunctionsToPlan(::substrait::Plan* plan) const;

  /// add extension types to Substrait plan.
  void addExtensionTypesToPlan(::substrait::Plan* plan) const;

  /// the count of extension function reference in a substrait plan.
  int functionReference_ = -1;
  /// extension function collected in substrait plan.
  std::shared_ptr<BiDirectionHashMap<FunctionId>> extensionFunctions_;
  /// extension function collected in substrait plan.
  std::shared_ptr<BiDirectionHashMap<TypeId>> extensionTypes_;

  ExtensionIdResolverPtr extensionIdResolver_;
};

using SubstraitExtensionCollectorPtr =
    std::shared_ptr<SubstraitExtensionCollector>;

} // namespace facebook::velox::substrait

namespace std {

/// hash function of facebook::velox::substrait::FunctionAnchor
template <>
struct hash<facebook::velox::substrait::FunctionId> {
  size_t operator()(const facebook::velox::substrait::FunctionId& k) const {
    size_t val = hash<std::string>()(k.uri);
    val = val * 31 + hash<std::string>()(k.signature);
    return val;
  }
};

/// hash function of facebook::velox::substrait::TypeId
template <>
struct hash<facebook::velox::substrait::TypeId> {
  size_t operator()(const facebook::velox::substrait::TypeId& k) const {
    size_t val = hash<std::string>()(k.uri);
    val = val * 31 + hash<std::string>()(k.name);
    return val;
  }
};

}; // namespace std