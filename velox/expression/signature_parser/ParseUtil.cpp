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

#include "velox/expression/signature_parser/ParseUtil.h"
#include <string>
#include "velox/type/Type.h"

namespace facebook::velox::exec {

TypeSignaturePtr inferTypeWithSpaces(
    std::vector<std::string>& words,
    bool cannotHaveFieldName) {
  VELOX_CHECK_GE(words.size(), 2);
  std::string fieldName = words[0];
  std::string typeName = words[1];
  for (int i = 2; i < words.size(); ++i) {
    typeName = fmt::format("{} {}", typeName, words[i]);
  }
  auto allWords = fmt::format("{} {}", fieldName, typeName);
  if (hasType(allWords) || cannotHaveFieldName) {
    return std::make_shared<exec::TypeSignature>(
        exec::TypeSignature(allWords, {}));
  }
  return std::make_shared<exec::TypeSignature>(
      exec::TypeSignature(typeName, {}, fieldName));
}

} // namespace facebook::velox::exec
