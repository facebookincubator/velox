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
    const velox::TypePtr& vType,
    std::shared_ptr<::substrait::Type> sType) {
  switch (vType->kind()) {
    case velox::TypeKind::BOOLEAN: {
      sType->set_allocated_bool_(new ::substrait::Type_Boolean());
      break;
    }
    case velox::TypeKind::TINYINT: {
      sType->set_allocated_i8(new ::substrait::Type_I8());
      break;
    }
    case velox::TypeKind::SMALLINT: {
      sType->set_allocated_i16(new ::substrait::Type_I16());
      break;
    }
    case velox::TypeKind::INTEGER: {
      sType->set_allocated_i32(new ::substrait::Type_I32());
      break;
    }
    case velox::TypeKind::BIGINT: {
      sType->set_allocated_i64(new ::substrait::Type_I64());
      break;
    }
    case velox::TypeKind::REAL: {
      sType->set_allocated_fp32(new ::substrait::Type_FP32());
      break;
    }
    case velox::TypeKind::DOUBLE: {
      sType->set_allocated_fp64(new ::substrait::Type_FP64());
      break;
    }
    case velox::TypeKind::VARCHAR: {
      sType->set_allocated_varchar(new ::substrait::Type_VarChar());
      break;
    }
    case velox::TypeKind::VARBINARY: {
      sType->set_allocated_binary(new ::substrait::Type_Binary());
      break;
    }
    case velox::TypeKind::TIMESTAMP: {
      sType->set_allocated_timestamp(new ::substrait::Type_Timestamp());
      break;
    }
    case velox::TypeKind::ARRAY: {
      ::substrait::Type_List* sTList = new ::substrait::Type_List();
      const TypePtr vElementType = vType->asArray().elementType();

      toSubstraitType(
          vElementType,
          std::make_shared<::substrait::Type>(*(sTList->mutable_type())));

      sType->set_allocated_list(sTList);
      break;
    }
    case velox::TypeKind::MAP: {
      ::substrait::Type_Map* sMap = new ::substrait::Type_Map();
      const TypePtr& vMapKeyType = vType->asMap().keyType();
      const TypePtr& vMapValueType = vType->asMap().valueType();

      toSubstraitType(
          vMapKeyType,
          std::make_shared<::substrait::Type>(*(sMap->mutable_key())));
      toSubstraitType(
          vMapValueType,
          std::make_shared<::substrait::Type>(*(sMap->mutable_value())));

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
    const velox::RowTypePtr& vRow,
    std::shared_ptr<::substrait::NamedStruct> sNamedStruct) {
  const int64_t vSize = vRow->size();
  const std::vector<std::string>& vNames = vRow->names();
  const std::vector<TypePtr>& vTypes = vRow->children();
  auto sTypeStruct = std::make_shared<::substrait::Type_Struct>(
      *(sNamedStruct->mutable_struct_()));

  for (int64_t i = 0; i < vSize; ++i) {
    const std::string& vName = vNames.at(i);
    const TypePtr& vType = vTypes.at(i);
    sNamedStruct->add_names(vName);

    toSubstraitType(
        vType,
        std::make_shared<::substrait::Type>(*(sTypeStruct->add_types())));
  }
}

} // namespace facebook::velox::substrait
