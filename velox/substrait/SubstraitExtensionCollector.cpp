/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "SubstraitExtensionCollector.h"
#include "velox/substrait/proto/substrait/extensions/extensions.pb.h"

namespace facebook::velox::substrait {

SubstraitExtensionCollector::SubstraitExtensionCollector() {
  functions_ = std::make_shared<BidiMap<SubstraitFunctionAnchor>>();
  types_ = std::make_shared<BidiMap<SubstraitTypeAnchor>>();
}

void SubstraitExtensionCollector::addFunctionToPlan(
    ::substrait::Plan* substraitPlan) const {
  using SimpleExtensionURI = ::substrait::extensions::SimpleExtensionURI;
  int uriPos = 1;
  std::unordered_map<std::string, SimpleExtensionURI*> uris;
  for (auto& [referenceNum, function] : functions_->forwardMap_) {
    SimpleExtensionURI* extensionUri;
    if (uris.find(function.uri) == uris.end()) {
      extensionUri = substraitPlan->add_extension_uris();
      extensionUri->set_extension_uri_anchor(++uriPos);
      extensionUri->set_uri(function.uri);
      uris[function.uri] = extensionUri;
    } else {
      extensionUri = uris.at(function.uri);
    }

    auto extensionFunction =
        substraitPlan->add_extensions()->mutable_extension_function();
    extensionFunction->set_extension_uri_reference(
        extensionUri->extension_uri_anchor());
    extensionFunction->set_function_anchor(referenceNum);
    extensionFunction->set_name(function.key);
  }
}

int SubstraitExtensionCollector::getFunctionReference(
    const SubstraitFunctionVariantPtr& function) {
  if (functions_->reverseMap_.find(function->anchor()) !=
      functions_->reverseMap_.end()) {
    return functions_->reverseMap_.at(function->anchor());
  }
  ++functionReference_;
  functions_->put(functionReference_, function->anchor());
  return functionReference_;
}

template <typename T>
void SubstraitExtensionCollector::BidiMap<T>::put(
    const int& key,
    const T& value) {
  forwardMap_[key] = value;
  reverseMap_[value] = key;
}

void SubstraitExtensionCollector::addExtensionToPlan(
    ::substrait::Plan* substraitPlan) const {
  addFunctionToPlan(substraitPlan);
  addTypeToPlan(substraitPlan);
}

void SubstraitExtensionCollector::addTypeToPlan(
    ::substrait::Plan* substraitPlan) const {
  using SimpleExtensionURI = ::substrait::extensions::SimpleExtensionURI;
  int uriPos = 1;
  std::unordered_map<std::string, SimpleExtensionURI*> uris;
  for (auto& [referenceNum, typeAnchor] : types_->forwardMap_) {
    SimpleExtensionURI* extensionUri;
    if (uris.find(typeAnchor.uri) == uris.end()) {
      extensionUri = substraitPlan->add_extension_uris();
      extensionUri->set_extension_uri_anchor(++uriPos);
      extensionUri->set_uri(typeAnchor.uri);
      uris[typeAnchor.uri] = extensionUri;
    } else {
      extensionUri = uris.at(typeAnchor.uri);
    }

    auto extensionType =
        substraitPlan->add_extensions()->mutable_extension_type();
    extensionType->set_extension_uri_reference(
        extensionUri->extension_uri_anchor());
    extensionType->set_type_anchor(referenceNum);
    extensionType->set_name(typeAnchor.name);
  }
}

int SubstraitExtensionCollector::getTypeReference(
    const SubstraitTypeAnchorPtr& typeAnchor) {
  if (types_->reverseMap_.find(*typeAnchor) != types_->reverseMap_.end()) {
    return types_->reverseMap_.at(*typeAnchor);
  }
  ++typeReference_;
  types_->put(functionReference_, *typeAnchor);
  return typeReference_;
}

} // namespace facebook::velox::substrait
