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

#include "velox/expression/FunctionSignature.h"
#include "velox/type/Type.h"

namespace facebook::velox::substrait {

class VeloxSubstraitSignature {
 public:
  /// Given a velox type name, return the Substrait type signature, throw if no
  /// match found.
  static std::string toSubstraitSignature(
      const exec::TypeSignature& typeSignature);

  /// Given a velox function name and argument types, return a matching function
  /// signature, throw if no match found.
  static const exec::FunctionSignature& resolveFunction(
      const std::string& functionName,
      const std::vector<TypePtr>& arguments);

  /// Given a velox function name and argument types, return the substrait
  /// function signature.
  static std::string toSubstraitSignature(
      const std::string& functionName,
      const std::vector<TypePtr>& arguments);

  /// Given a velox function name and argument types, return the velox function
  /// signature.
  static std::string toVeloxSignature(
      const std::string& functionName,
      const std::vector<facebook::velox::TypePtr>& inputs);

  /// Given a collection of function signature, return the velox function
  /// signatures join by the ',' delimiter.
  static std::string toVeloxSignature(
      const std::vector<const facebook::velox::exec::FunctionSignature*>&
          signatures);
};

} // namespace facebook::velox::substrait
