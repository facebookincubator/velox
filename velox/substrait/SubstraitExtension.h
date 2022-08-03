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

#include "SubstraitFunction.h"
#include "velox/common/base/Exceptions.h"
#include "velox/substrait/SubstraitFunction.h"

namespace facebook::velox::substrait {

// class used to deserialize substrait YAML extension files.
class SubstraitExtension {
 public:
  SubstraitExtension() {}

  //
  static std::shared_ptr<SubstraitExtension> load() {
    static SubstraitExtension substraitExtension;
    return std::make_shared<SubstraitExtension>(substraitExtension);
  }

  const std::vector<SubstraitFunctionPtr> scalarFunctions() const {
    return scalarFunctions_;
  }
  const std::vector<SubstraitFunctionPtr> aggregateFunctions() const {
    return aggregateFunctions_;
  }
  const std::vector<SubstraitFunctionPtr> windowFunctions() const {
    return windowFunctions_;
  }

 private:
  std::vector<SubstraitFunctionPtr> scalarFunctions_;
  std::vector<SubstraitFunctionPtr> aggregateFunctions_;
  std::vector<SubstraitFunctionPtr> windowFunctions_;
};

using SubstraitExtensionPtr = std::shared_ptr<const SubstraitExtension>;

} // namespace facebook::velox::substrait
