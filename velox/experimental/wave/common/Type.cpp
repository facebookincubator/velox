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

#include "velox/experimental/wave/common/Type.h"

#include "velox/type/Type.h"

namespace facebook::velox::wave {

PhysicalType fromCpuType(const Type& type) {
  PhysicalType ans{};
  switch (type.kind()) {
    case TypeKind::TINYINT:
      ans.kind = PhysicalType::kInt8;
      break;
    case TypeKind::SMALLINT:
      ans.kind = PhysicalType::kInt16;
      break;
    case TypeKind::INTEGER:
      ans.kind = PhysicalType::kInt32;
      break;
    case TypeKind::BIGINT:
      ans.kind = PhysicalType::kInt64;
      break;
    case TypeKind::REAL:
      ans.kind = PhysicalType::kFloat32;
      break;
    case TypeKind::DOUBLE:
      ans.kind = PhysicalType::kFloat64;
      break;
    case TypeKind::VARCHAR:
      ans.kind = PhysicalType::kString;
      break;
    default:
      VELOX_UNSUPPORTED("{}", type.kind());
  }
  return ans;
}

std::string_view PhysicalType::kindString(Kind kind) {
  switch (kind) {
    case kInt8:
      return "Int8";
    case kInt16:
      return "Int16";
    case kInt32:
      return "Int32";
    case kInt64:
      return "Int64";
    case kInt128:
      return "Int128";
    case kFloat32:
      return "Float32";
    case kFloat64:
      return "Float64";
    case kString:
      return "String";
    case kArray:
      return "Array";
    case kMap:
      return "Map";
    case kRow:
      return "Row";
  }

  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::wave
