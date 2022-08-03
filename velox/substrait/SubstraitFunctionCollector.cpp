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

#include "SubstraitFunctionCollector.h"
#include "velox/substrait/SubstraitFunction.h"
#include "velox/substrait/proto/substrait/extensions/extensions.pb.h"

namespace facebook::velox::substrait {

const void SubstraitFunctionCollector::addFunctionToPlan(
    ::substrait::Plan& substraitPlan) const {
  int uriPos = 1;

  auto uris = std::unordered_map<
      std::string,
      ::substrait::extensions::SimpleExtensionURI*>{};

  for (auto& [referenceNum, function] : funcMap_->forwardMap_) {
    ::substrait::extensions::SimpleExtensionURI* extensionUri;
    if (uris.find(function->uri) == uris.end()) {
      extensionUri = substraitPlan.add_extension_uris();
      extensionUri->set_extension_uri_anchor(++uriPos);
      extensionUri->set_uri(function->uri);
      uris[function->uri] = extensionUri;
    } else {
      extensionUri = uris.at(function->uri);
    }

    auto extensionFunction =
        substraitPlan.add_extensions()->mutable_extension_function();
    extensionFunction->set_extension_uri_reference(
        extensionUri->extension_uri_anchor());
    extensionFunction->set_function_anchor(referenceNum);
    extensionFunction->set_name(function->key());
  }
}

const int SubstraitFunctionCollector::getFunctionReference(
    const SubstraitFunctionPtr& function) {
  if (funcMap_->reverseMap_.end() != funcMap_->reverseMap_.end()) {
    return funcMap_->reverseMap_.at(function);
  }

  ++counter_;
  funcMap_->put(counter_, function);
  return counter_;
}

void SubstraitFunctionCollector::BidiMap::put(
    int reference,
    SubstraitFunctionPtr function) {
  forwardMap_[reference] = function;
  reverseMap_[function] = reference;
}
} // namespace facebook::velox::substrait
