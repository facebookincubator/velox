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

#include "SubstraitType.h"

namespace facebook::velox::substrait {

const std::string SubstraitTypeUtil::typeToSignature(
    const ::substrait::Type& type) {
  if (type.kind_case() == substrait::SubstraitTypeKind::kUserDefined) {
    return ANY_TYPE->signature();
  } else {
    if (TYPES.find(type.kind_case()) != TYPES.end()) {
      return TYPES.at(type.kind_case())->signature();
    } else {
      VELOX_NYI(
          "Returning Substrait signature of Substrait Type not supported for Substrait type {}.",
          type.kind_case());
    }
  }
}

const SubstraitTypePtr SubstraitTypeUtil::parseType(
    const std::string& rawType) {
  std::string lowerCaseRawType = rawType;
  std::transform(
      lowerCaseRawType.begin(), lowerCaseRawType.end(), lowerCaseRawType.begin(), [](unsigned char c) {
        return std::tolower(c);
      });

  if (lowerCaseRawType.rfind(ANY_TYPE->rawType(), 0) == 0) {
    return ANY_TYPE;
  } else if (lowerCaseRawType.rfind(UNKNOWN_TYPE->rawType(), 0) == 0) {
    return UNKNOWN_TYPE;
  } else {
    for (auto& [typeKind, type] : TYPES) {
      if (lowerCaseRawType.rfind(type->rawType(), 0) == 0) {
        return type;
      }
    }
    VELOX_NYI(
        "Returning Substrait Type not supported for raw type {}.", rawType);
  }
}

std::string SubstraitTypeUtil::signature(
    const std::string& functionName,
    const std::vector<::substrait::Type>& substraitTypes) {
  std::stringstream signature;
  signature << functionName << ":";
  for (auto it = substraitTypes.begin(); it != substraitTypes.end(); ++it) {
    const auto& typeSign = typeToSignature(*it);
    if (it == substraitTypes.end() - 1) {
      signature << typeSign;
    } else {
      signature << typeSign << "_";
    }
  }
  return signature.str();
}

} // namespace facebook::velox::substrait
