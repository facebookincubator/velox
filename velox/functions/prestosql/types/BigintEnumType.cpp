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

#include "velox/functions/prestosql/types/BigintEnumType.h"

namespace facebook::velox {

// Should only be called from get() to create a new instance.
BigintEnumType::BigintEnumType(const LongEnumParameter& parameters)
    : EnumTypeBase<int64_t, LongEnumParameter, BigintType>(parameters) {}

std::string BigintEnumType::toString() const {
  return fmt::format("{}:BigintEnum({})", name_, flippedMapToString());
}

BigintEnumTypePtr BigintEnumType::get(const LongEnumParameter& parameter) {
  return getCached<BigintEnumType>(parameter);
}

folly::dynamic BigintEnumType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = name();
  // parameters_[0].longEnumLiteral is assumed to have a value since it is
  // constructed from a LongEnumParameter.
  obj["kLongEnumParam"] = parameters_[0].longEnumLiteral.value().serialize();
  return obj;
}

} // namespace facebook::velox
