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

int SubstraitExtensionCollector::getFunctionReference(
    const std::string& functionName,
    const std::vector<TypePtr>& arguments) {
  const auto& functionId =
      extensionIdResolver_->resolve(functionName, arguments);
  const auto& anchorReference =
      extensionFunctions_->reverseMap_.find(functionId);
  if (anchorReference != extensionFunctions_->reverseMap_.end()) {
    return anchorReference->second;
  }
  ++functionReference_;
  extensionFunctions_->put(functionReference_, functionId);
  return functionReference_;
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

FunctionId ExtensionIdResolver::resolve(
    const std::string& funcName,
    const std::vector<TypePtr>& arguments) const {
  if (arguments.empty()) {
    return {"", funcName};
  }
  std::vector<std::string> typeSignature;
  typeSignature.reserve(arguments.size());
  for (const auto& arg : arguments) {
    typeSignature.emplace_back(toSubstraitSignature(arg));
  }
  std::string signature = funcName + ":" + folly::join("_", typeSignature);
  // TODO: currently we just treat the engine-own function signatures as
  // substrait extension function signature.
  return {"", signature};
}

} // namespace facebook::velox::substrait
