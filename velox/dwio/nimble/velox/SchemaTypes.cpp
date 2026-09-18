/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/velox/SchemaTypes.h"

#include "velox/dwio/nimble/common/Exceptions.h"

#include <fmt/format.h>

namespace facebook::nimble {
std::string toString(ScalarKind kind) {
  switch (kind) {
#define CASE(KIND)         \
  case ScalarKind::KIND: { \
    return #KIND;          \
  }
    CASE(Int8);
    CASE(UInt8);
    CASE(Int16);
    CASE(UInt16);
    CASE(Int32);
    CASE(UInt32);
    CASE(Int64);
    CASE(UInt64);
    CASE(Float);
    CASE(Double);
    CASE(Bool);
    CASE(String);
    CASE(Binary);
    CASE(Undefined);
#undef CASE
  }
  NIMBLE_UNREACHABLE("Unknown: {}.", static_cast<int32_t>(kind));
}

std::string toString(Kind kind) {
  switch (kind) {
#define CASE(KIND)   \
  case Kind::KIND: { \
    return #KIND;    \
  }
    CASE(Scalar);
    CASE(TimestampMicroNano);
    CASE(Row);
    CASE(Array);
    CASE(ArrayWithOffsets);
    CASE(Map);
    CASE(FlatMap);
    CASE(SlidingWindowMap);
#undef CASE
  }
  NIMBLE_UNREACHABLE("Unknown: {}.", static_cast<int32_t>(kind));
}

} // namespace facebook::nimble
