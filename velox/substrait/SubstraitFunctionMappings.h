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

#include "velox/common/base/Exceptions.h"
#include "velox/substrait/SubstraitParser.h"

namespace facebook::velox::substrait {

// A map maintain function names in difference between Velox and Substrait
// key: velox function name
// value: substrait function
using FunctionMappingMap = std::unordered_map<std::string, std::string>;


struct SubstraitFunctionMappings {
  //scalar function names in difference between velox and Substrait
  static const FunctionMappingMap scalarMappings() {
   static FunctionMappingMap scalarMappings;
   return scalarMappings;
  };

  //aggregate function names in difference between velox and Substrait
  static const FunctionMappingMap aggregateMappings() {
    static FunctionMappingMap aggregateMappings;
    return aggregateMappings;
  }

  //window function names in difference between velox and Substrait
  static const FunctionMappingMap windowMappings() {
    static FunctionMappingMap windowMappings;
    return windowMappings;
  }

};

} // namespace facebook::velox::substrait
