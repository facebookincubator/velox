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

::substrait::Type VeloxToSubstraitTypeConvertor::toSubstraitType(
    const velox::TypePtr& vType,
    ::substrait::Type* sType) {
  switch (vType->kind()) {
    case velox::TypeKind::BOOLEAN: {
      sType->set_allocated_bool_(new ::substrait::Type_Boolean());
      return *sType;
    }
    case velox::TypeKind::TINYINT: {
      sType->set_allocated_i8(new ::substrait::Type_I8());
      return *sType;
    }
    case velox::TypeKind::SMALLINT: {
      sType->set_allocated_i16(new ::substrait::Type_I16());
      return *sType;
    }
    case velox::TypeKind::INTEGER: {
      sType->set_allocated_i32(new ::substrait::Type_I32());
      return *sType;
    }
    case velox::TypeKind::BIGINT: {
      sType->set_allocated_i64(new ::substrait::Type_I64());
      return *sType;
    }
    case velox::TypeKind::REAL: {
      sType->set_allocated_fp32(new ::substrait::Type_FP32());
      return *sType;
    }
    case velox::TypeKind::DOUBLE: {
      sType->set_allocated_fp64(new ::substrait::Type_FP64());
      return *sType;
    }
    case velox::TypeKind::VARCHAR: {
      sType->set_allocated_varchar(new ::substrait::Type_VarChar());
      return *sType;
    }
    case velox::TypeKind::VARBINARY: {
      sType->set_allocated_binary(new ::substrait::Type_Binary());
      return *sType;
    }
    case velox::TypeKind::TIMESTAMP: {
      sType->set_allocated_timestamp(new ::substrait::Type_Timestamp());
      return *sType;
    }
    case velox::TypeKind::ARRAY: {
      ::substrait::Type_List* sTList = new ::substrait::Type_List();
      const std::shared_ptr<const Type> vArrayType =
          vType->asArray().elementType();
      ::substrait::Type sListType =
          toSubstraitType(vArrayType, sTList->mutable_type());

      sType->set_allocated_list(sTList);
      return *sType;
    }
    case velox::TypeKind::MAP: {
      ::substrait::Type_Map* sMap = new ::substrait::Type_Map();
      const std::shared_ptr<const Type> vMapKeyType = vType->asMap().keyType();
      const std::shared_ptr<const Type> vMapValueType =
          vType->asMap().valueType();

      toSubstraitType(vMapKeyType, sMap->mutable_key());
      toSubstraitType(vMapValueType, sMap->mutable_value());

      sType->set_allocated_map(sMap);
      return *sType;
    }
    case velox::TypeKind::UNKNOWN:
    case velox::TypeKind::FUNCTION:
    case velox::TypeKind::OPAQUE:
    case velox::TypeKind::INVALID:
    default:
      throw std::runtime_error(
          "Unsupported type " + std::string(vType->kindName()));
  }
}

::substrait::NamedStruct* VeloxToSubstraitTypeConvertor::toSubstraitNamedStruct(
    const velox::RowTypePtr& vRow,
    ::substrait::NamedStruct* sNamedStruct) {
  int64_t vSize = vRow->size();
  std::vector<std::string> vNames = vRow->names();
  std::vector<std::shared_ptr<const Type>> vTypes = vRow->children();

  for (int64_t i = 0; i < vSize; ++i) {
    std::string vName = vNames.at(i);
    std::shared_ptr<const Type> vType = vTypes.at(i);
    sNamedStruct->add_names(vName);
    ::substrait::Type* sStruct = sNamedStruct->mutable_struct_()->add_types();

    toSubstraitType(vType, sStruct);
  }

  return sNamedStruct;
}

::substrait::Expression_Literal*
VeloxToSubstraitTypeConvertor::processVeloxValueByType(
    ::substrait::Expression_Literal_Struct* sLitValue,
    ::substrait::Expression_Literal* sField,
    const velox::VectorPtr& children) {
  // to handle the null value. TODO need to confirm
  std::optional<vector_size_t> nullCount = children->getNullCount();
  // should be the same with rowValue->type();
  std::shared_ptr<const Type> childType = children->type();
  switch (childType->kind()) {
    case velox::TypeKind::BOOLEAN: {
      auto childToFlatVec = children->asFlatVector<bool>();
      vector_size_t flatVecSzie = childToFlatVec->size();
      if (nullCount.has_value() && nullCount != 0) {
        sField = processVeloxNullValueByCount(
            childType, nullCount, sLitValue, sField);
      } else {
        for (int64_t i = 0; i < flatVecSzie; i++) {
          sField = sLitValue->add_fields();
          sField->set_boolean(childToFlatVec->valueAt(i));
        }
      }
      return sField;
    }
    case velox::TypeKind::TINYINT: {
      auto childToFlatVec = children->asFlatVector<int8_t>();
      vector_size_t flatVecSzie = childToFlatVec->size();

      if (nullCount.has_value() && nullCount != 0) {
        sField = processVeloxNullValueByCount(
            childType, nullCount, sLitValue, sField);
      } else {
        for (int64_t i = 0; i < flatVecSzie; i++) {
          sField = sLitValue->add_fields();
          sField->set_i8(childToFlatVec->valueAt(i));
        }
      }
      return sField;
    }
    case velox::TypeKind::SMALLINT: {
      auto childToFlatVec = children->asFlatVector<int16_t>();
      vector_size_t flatVecSzie = childToFlatVec->size();
      if (nullCount.has_value() && nullCount != 0) {
        sField = processVeloxNullValueByCount(
            childType, nullCount, sLitValue, sField);
      } else {
        for (int64_t i = 0; i < flatVecSzie; i++) {
          sField = sLitValue->add_fields();
          sField->set_i16(childToFlatVec->valueAt(i));
        }
      }
      return sField;
    }
    case velox::TypeKind::INTEGER: {
      if (nullCount.has_value() && nullCount != 0) {
        sField = processVeloxNullValueByCount(
            childType, nullCount, sLitValue, sField);
      } else {
        auto childToFlatVec = children->asFlatVector<int32_t>();
        vector_size_t flatVecSzie = childToFlatVec->size();
        for (int64_t i = 0; i < flatVecSzie; i++) {
          sField = sLitValue->add_fields();
          sField->set_i32(childToFlatVec->valueAt(i));
        }
      }
      return sField;
    }
    case velox::TypeKind::BIGINT: {
      auto childToFlatVec = children->asFlatVector<int64_t>();
      vector_size_t flatVecSzie = childToFlatVec->size();
      if (nullCount.has_value() && nullCount != 0) {
        sField = processVeloxNullValueByCount(
            childType, nullCount, sLitValue, sField);
      } else {
        for (int64_t i = 0; i < flatVecSzie; i++) {
          sField = sLitValue->add_fields();
          sField->set_i64(childToFlatVec->valueAt(i));
        }
      }
      return sField;
    }
    case velox::TypeKind::REAL: {
      auto childToFlatVec = children->asFlatVector<float_t>();
      vector_size_t flatVecSzie = childToFlatVec->size();
      if (nullCount.has_value() && nullCount != 0) {
        sField = processVeloxNullValueByCount(
            childType, nullCount, sLitValue, sField);
      } else {
        for (int64_t i = 0; i < flatVecSzie; i++) {
          sField = sLitValue->add_fields();
          sField->set_fp32(childToFlatVec->valueAt(i));
        }
      }
      return sField;
    }
    case velox::TypeKind::DOUBLE: {
      auto childToFlatVec = children->asFlatVector<double_t>();
      vector_size_t flatVecSzie = childToFlatVec->size();
      if (nullCount.has_value() && nullCount != 0) {
        sField = processVeloxNullValueByCount(
            childType, nullCount, sLitValue, sField);
      } else {
        for (int64_t i = 0; i < flatVecSzie; i++) {
          sField = sLitValue->add_fields();
          sField->set_fp64(childToFlatVec->valueAt(i));
        }
      }
      return sField;
    }
    case velox::TypeKind::VARCHAR: {
      auto childToFlatVec = children->asFlatVector<StringView>();
      vector_size_t flatVecSzie = childToFlatVec->size();
      if (nullCount.has_value() && nullCount != 0) {
        auto tmp0 = children->type();
        sField = processVeloxNullValueByCount(
            childType, nullCount, sLitValue, sField);
      } else {
        for (int64_t i = 0; i < flatVecSzie; i++) {
          sField = sLitValue->add_fields();
          ::substrait::Expression_Literal::VarChar* sVarChar =
              new ::substrait::Expression_Literal::VarChar();
          StringView vChildValueAt = childToFlatVec->valueAt(i);
          sVarChar->set_value(vChildValueAt);
          sVarChar->set_length(vChildValueAt.size());
          sField->set_allocated_var_char(sVarChar);
        }
      }
      return sField;
    }
    default:
      throw std::runtime_error(
          "Unsupported type " + std::string(childType->kindName()));
  }
}

::substrait::Expression_Literal*
VeloxToSubstraitTypeConvertor::processVeloxNullValueByCount(
    const velox::TypePtr& childType,
    std::optional<vector_size_t> nullCount,
    ::substrait::Expression_Literal_Struct* sLitValue,
    ::substrait::Expression_Literal* sField) {
  for (int64_t i = 0; i < nullCount.value(); i++) {
    sField = sLitValue->add_fields();
    processVeloxNullValue(sField, childType);
  }
  return sField;
}

::substrait::Expression_Literal*
VeloxToSubstraitTypeConvertor::processVeloxNullValue(
    ::substrait::Expression_Literal* sField,
    const velox::TypePtr& childType) {
  switch (childType->kind()) {
    case velox::TypeKind::BOOLEAN: {
      ::substrait::Type_Boolean* nullValue = new ::substrait::Type_Boolean();
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_bool_(nullValue);
      break;
    }
    case velox::TypeKind::TINYINT: {
      ::substrait::Type_I8* nullValue = new ::substrait::Type_I8();
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_i8(nullValue);
      break;
    }
    case velox::TypeKind::SMALLINT: {
      ::substrait::Type_I16* nullValue = new ::substrait::Type_I16();
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_i16(nullValue);
      break;
    }
    case velox::TypeKind::INTEGER: {
      ::substrait::Type_I32* nullValue = new ::substrait::Type_I32();
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_i32(nullValue);
      break;
    }
    case velox::TypeKind::BIGINT: {
      ::substrait::Type_I64* nullValue = new ::substrait::Type_I64();
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_i64(nullValue);
      break;
    }
    case velox::TypeKind::VARCHAR: {
      ::substrait::Type_VarChar* nullValue = new ::substrait::Type_VarChar();
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_varchar(nullValue);
      break;
    }
    case velox::TypeKind::REAL: {
      ::substrait::Type_FP32* nullValue = new ::substrait::Type_FP32();
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_fp32(nullValue);
      break;
    }
    case velox::TypeKind::DOUBLE: {
      ::substrait::Type_FP64* nullValue = new ::substrait::Type_FP64();
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_fp64(nullValue);
      break;
    }
    default: {
      throw std::runtime_error(
          "Unsupported type " + std::string(childType->kindName()));
    }
  }

  return sField;
}

} // namespace facebook::velox::substrait
