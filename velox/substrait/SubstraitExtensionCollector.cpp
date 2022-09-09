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

#include "velox/substrait/SubstraitExtensionCollector.h"
#include "velox/substrait/TypeUtils.h"

namespace facebook::velox::substrait {

SubstraitExtensionCollector::SubstraitExtensionCollector() {
  extensionFunctions_ = std::make_shared<BiDirectionHashMap<FunctionId>>();
  extensionIdResolver_ = std::make_shared<ExtensionIdResolver>();
}

void SubstraitExtensionCollector::addExtensionFunctionsToPlan(
    ::substrait::Plan* plan) const {
  using SimpleExtensionURI = ::substrait::extensions::SimpleExtensionURI;
  int uriPos = 1;
  std::unordered_map<std::string, SimpleExtensionURI*> uris;
  for (auto& [referenceNum, functionId] : extensionFunctions_->forwardMap_) {
    SimpleExtensionURI* extensionUri;
    const auto uri = uris.find(functionId.uri);
    if (uri == uris.end()) {
      extensionUri = plan->add_extension_uris();
      extensionUri->set_extension_uri_anchor(++uriPos);
      extensionUri->set_uri(functionId.uri);
      uris[functionId.uri] = extensionUri;
    } else {
      extensionUri = uri->second;
    }

    auto extensionFunction =
        plan->add_extensions()->mutable_extension_function();
    extensionFunction->set_extension_uri_reference(
        extensionUri->extension_uri_anchor());
    extensionFunction->set_function_anchor(referenceNum);
    extensionFunction->set_name(functionId.signature);
  }
}


void SubstraitExtensionCollector::addExtensionTypesToPlan(
    ::substrait::Plan* plan) const {
  using SimpleExtensionURI = ::substrait::extensions::SimpleExtensionURI;
  int uriPos = 1;
  std::unordered_map<std::string, SimpleExtensionURI*> uris;
  for (auto& [referenceNum, typeId] : extensionTypes_->forwardMap_) {
    SimpleExtensionURI* extensionUri;
    const auto uri = uris.find(typeId.uri);
    if (uri == uris.end()) {
      extensionUri = plan->add_extension_uris();
      extensionUri->set_extension_uri_anchor(++uriPos);
      extensionUri->set_uri(typeId.uri);
      uris[typeId.uri] = extensionUri;
    } else {
      extensionUri = uri->second;
    }

    auto extensionType =
        plan->add_extensions()->mutable_extension_type();
    extensionType->set_extension_uri_reference(
        extensionUri->extension_uri_anchor());
    extensionType->set_type_anchor(referenceNum);
    extensionType->set_name(typeId.name);
  }
}

std::optional<int> SubstraitExtensionCollector::getFunctionReference(
    const std::string& functionName,
    const std::vector<TypePtr>& arguments) {
  const auto& functionId =
      extensionIdResolver_->resolve(functionName, arguments);
  const auto& anchorReference =
      extensionFunctions_->reverseMap_.find(functionId);
  if (anchorReference != extensionFunctions_->reverseMap_.end()) {
    return std::make_optional(anchorReference->second);
  }
  ++functionReference_;
  extensionFunctions_->put(functionReference_, functionId);
  return std::make_optional(functionReference_);
}

template <typename T>
void SubstraitExtensionCollector::BiDirectionHashMap<T>::put(
    const int& key,
    const T& value) {
  forwardMap_[key] = value;
  reverseMap_[value] = key;
}

void SubstraitExtensionCollector::addExtensionsToPlan(
    ::substrait::Plan* plan) const {
  addExtensionFunctionsToPlan(plan);
}
std::optional<int> SubstraitExtensionCollector::getTypeReference(
    const TypePtr& type) {
  const auto& typeId = extensionIdResolver_->resolve(type);
  const auto& typeReference = extensionTypes_->reverseMap_.find(typeId);
  if (typeReference != extensionTypes_->reverseMap_.end()) {
    return std::make_optional(typeReference->second);
  }
  ++functionReference_;
  extensionTypes_->put(functionReference_, typeId);
  return std::make_optional(functionReference_);
}

FunctionId ExtensionIdResolver::resolve(
    const std::string& funcName,
    const std::vector<TypePtr>& arguments) const {
  if (!arguments.empty()) {
    std::vector<std::string> typeSignature;
    typeSignature.reserve(arguments.size());
    for (const auto& arg : arguments) {
      typeSignature.emplace_back(substraitSignature(arg));
    }
    std::string signature = funcName + ":" + folly::join("_", typeSignature);
    return {"", signature};
  }
  return {"", funcName};
}

TypeId ExtensionIdResolver::resolve(const TypePtr& type) const {
  VELOX_CHECK(type->isUnKnown(), "currently only support velox unknown type");
  return TypeId{"", "UNKNOWN"};
}

} // namespace facebook::velox::substrait
