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

namespace facebook::velox::substrait {

namespace {
std::string toString(
    const std::string& functionName,
    const std::vector<facebook::velox::TypePtr>& inputs) {
  std::ostringstream signature;
  signature << functionName << "(";
  for (auto i = 0; i < inputs.size(); i++) {
    if (i > 0) {
      signature << ", ";
    }
    signature << inputs[i]->toString();
  }
  signature << ")";
  return signature.str();
}

std::string toString(
    const std::vector<const facebook::velox::exec::FunctionSignature*>&
        signatures) {
  std::stringstream out;
  for (auto i = 0; i < signatures.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << signatures[i]->toString();
  }
  return out.str();
}

} // namespace

int SubstraitExtensionCollector::getReferenceNumber(
    const std::string& functionName,
    const std::vector<TypePtr>& arguments) {
  const auto& extensionFunctionId =
      ExtensionFunctionId::create(functionName, arguments);
  const auto& extensionFunctionAnchorIt =
      extensionFunctions_->reverseMap_.find(extensionFunctionId);
  if (extensionFunctionAnchorIt != extensionFunctions_->reverseMap_.end()) {
    return extensionFunctionAnchorIt->second;
  }
  ++functionReference_;
  extensionFunctions_->put(functionReference_, extensionFunctionId);
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

SubstraitExtensionCollector::SubstraitExtensionCollector() {
  extensionFunctions_ =
      std::make_shared<BiDirectionHashMap<ExtensionFunctionId>>();
}

ExtensionFunctionId ExtensionFunctionId::create(
    const std::string& functionName,
    const std::vector<TypePtr>& arguments) {
  const auto& substraitFunctionSignature =
      VeloxSubstraitSignature::toSubstraitSignature(functionName, arguments);

  /// TODO: Currently we treat all velox registry based function signatures as
  /// custom substrait extension, so no uri link and leave it as empty.
  const std::string uri;
  return {uri, substraitFunctionSignature};
}

} // namespace facebook::velox::substrait
