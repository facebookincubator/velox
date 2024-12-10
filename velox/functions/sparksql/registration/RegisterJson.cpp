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
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/sparksql/JsonFunctions.h"
#include "velox/functions/sparksql/JsonObjectKeys.h"

namespace facebook::velox::functions::sparksql {

void registerJsonFunctions(const std::string& prefix) {
  registerFunction<GetJsonObjectFunction, Varchar, Varchar, Varchar>(
      {prefix + "get_json_object"});
  registerFunction<JsonObjectKeysFunction, Array<Varchar>, Varchar>(
      {prefix + "json_object_keys"});
}

} // namespace facebook::velox::functions::sparksql
