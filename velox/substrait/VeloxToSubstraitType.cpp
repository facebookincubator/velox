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

#include "velox/substrait/VeloxToSubstraitType.h"

#include "velox/expression/Expr.h"

namespace facebook::velox::substrait {

void VeloxToSubstraitTypeConvertor::toSubstraitType(
    google::protobuf::Arena& arena,
    const velox::TypePtr& vType,
    ::substrait::Type* sType) {
  switch (vType->kind()) {
    case velox::TypeKind::BOOLEAN: {
      sType->set_allocated_bool_(
          google::protobuf::Arena::CreateMessage<::substrait::Type_Boolean>(
              &arena));
      break;
    }
    case velox::TypeKind::TINYINT: {
      sType->set_allocated_i8(
          google::protobuf::Arena::CreateMessage<::substrait::Type_I8>(&arena));
      break;
    }
    case velox::TypeKind::SMALLINT: {
      sType->set_allocated_i16(
          google::protobuf::Arena::CreateMessage<::substrait::Type_I16>(
              &arena));
      break;
    }
    case velox::TypeKind::INTEGER: {
      sType->set_allocated_i32(
          google::protobuf::Arena::CreateMessage<::substrait::Type_I32>(
              &arena));
      break;
    }
    case velox::TypeKind::BIGINT: {
      sType->set_allocated_i64(
          google::protobuf::Arena::CreateMessage<::substrait::Type_I64>(
              &arena));
      break;
    }
    case velox::TypeKind::REAL: {
      sType->set_allocated_fp32(
          google::protobuf::Arena::CreateMessage<::substrait::Type_FP32>(
              &arena));
      break;
    }
    case velox::TypeKind::DOUBLE: {
      sType->set_allocated_fp64(
          google::protobuf::Arena::CreateMessage<::substrait::Type_FP64>(
              &arena));
      break;
    }
    case velox::TypeKind::VARCHAR: {
      sType->set_allocated_varchar(
          google::protobuf::Arena::CreateMessage<::substrait::Type_VarChar>(
              &arena));
      break;
    }
    case velox::TypeKind::VARBINARY: {
      sType->set_allocated_binary(
          google::protobuf::Arena::CreateMessage<::substrait::Type_Binary>(
              &arena));
      break;
    }
    case velox::TypeKind::TIMESTAMP: {
      sType->set_allocated_timestamp(
          google::protobuf::Arena::CreateMessage<::substrait::Type_Timestamp>(
              &arena));
      break;
    }
    case velox::TypeKind::ARRAY: {
      ::substrait::Type_List* sTList =
          google::protobuf::Arena::CreateMessage<::substrait::Type_List>(
              &arena);

      const TypePtr vElementType = vType->asArray().elementType();
      toSubstraitType(arena, vElementType, sTList->mutable_type());
      sType->set_allocated_list(sTList);

      break;
    }
    case velox::TypeKind::MAP: {
      ::substrait::Type_Map* sMap =
          google::protobuf::Arena::CreateMessage<::substrait::Type_Map>(&arena);

      const TypePtr& vMapKeyType = vType->asMap().keyType();
      const TypePtr& vMapValueType = vType->asMap().valueType();

      toSubstraitType(arena, vMapKeyType, sMap->mutable_key());
      toSubstraitType(arena, vMapValueType, sMap->mutable_value());

      sType->set_allocated_map(sMap);

      break;
    }
    case velox::TypeKind::UNKNOWN:
    case velox::TypeKind::FUNCTION:
    case velox::TypeKind::OPAQUE:
    case velox::TypeKind::INVALID:
    default:
      VELOX_UNSUPPORTED("Unsupported type '{}'", std::string(vType->kindName()))
  }
}

void VeloxToSubstraitTypeConvertor::toSubstraitNamedStruct(
    google::protobuf::Arena& arena,
    const velox::RowTypePtr& vRow,
    ::substrait::NamedStruct* sNamedStruct) {
  const int64_t vSize = vRow->size();
  const std::vector<std::string>& vNames = vRow->names();
  const std::vector<TypePtr>& vTypes = vRow->children();
  ::substrait::Type_Struct* sTypeStruct = sNamedStruct->mutable_struct_();

  for (int64_t i = 0; i < vSize; ++i) {
    const std::string& vName = vNames.at(i);
    const TypePtr& vType = vTypes.at(i);
    sNamedStruct->add_names(vName);

    toSubstraitType(arena, vType, sTypeStruct->add_types());
  }
}

} // namespace facebook::velox::substrait
