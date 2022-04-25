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

#include "velox/substrait/VeloxToSubstraitExpr.h"

#include "velox/expression/ControlExpr.h"

namespace facebook::velox::substrait {

::substrait::Expression* VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const ITypedExpr>& vExpr,
    const RowTypePtr& vPreNodeOutPut) {
  ::substrait::Expression* sExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression>(&arena);
  if (auto vConstExpr =
          std::dynamic_pointer_cast<const ConstantTypedExpr>(vExpr)) {
    sExpr->set_allocated_literal(
        toSubstraitExpr(arena, vConstExpr, vPreNodeOutPut));
    return sExpr;
  }
  if (auto vCallTypeExpr =
          std::dynamic_pointer_cast<const CallTypedExpr>(vExpr)) {
    sExpr->MergeFrom(*toSubstraitExpr(arena, vCallTypeExpr, vPreNodeOutPut));
    return sExpr;
  }
  if (auto vFieldExpr =
          std::dynamic_pointer_cast<const FieldAccessTypedExpr>(vExpr)) {
    sExpr->set_allocated_selection(
        toSubstraitExpr(arena, vFieldExpr, vPreNodeOutPut));
    return sExpr;
  }
  if (auto vCastExpr = std::dynamic_pointer_cast<const CastTypedExpr>(vExpr)) {
    sExpr->set_allocated_cast(
        toSubstraitExpr(arena, vCastExpr, vPreNodeOutPut));
    return sExpr;
  } else {
    VELOX_UNSUPPORTED("Unsupport Expr '{}' in Substrait", vExpr->toString());
  }
}

::substrait::Expression_Cast* VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const CastTypedExpr>& vCastExpr,
    const RowTypePtr& vPreNodeOutPut) {
  ::substrait::Expression_Cast* sCastExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression_Cast>(
          &arena);
  std::vector<std::shared_ptr<const ITypedExpr>> vCastTypeInputs =
      vCastExpr->inputs();

  v2STypeConvertor_.toSubstraitType(
      arena, vCastExpr->type(), sCastExpr->mutable_type());

  for (auto& vArg : vCastTypeInputs) {
    sCastExpr->set_allocated_input(
        toSubstraitExpr(arena, vArg, vPreNodeOutPut));
  }
  return sCastExpr;
}

::substrait::Expression_FieldReference*
VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const FieldAccessTypedExpr>& vFieldExpr,
    const RowTypePtr& vPreNodeOutPut) {
  ::substrait::Expression_FieldReference* sFieldExpr =
      google::protobuf::Arena::CreateMessage<
          ::substrait::Expression_FieldReference>(&arena);
  // kSelection
  const std::shared_ptr<const Type> vExprType = vFieldExpr->type();
  std::string vExprName = vFieldExpr->name();

  ::substrait::Expression_ReferenceSegment_StructField* sDirectStruct =
      sFieldExpr->mutable_direct_reference()->mutable_struct_field();

  std::vector<std::string> vPreNodeColNames = vPreNodeOutPut->names();
  std::vector<std::shared_ptr<const velox::Type>> vPreNodeColTypes =
      vPreNodeOutPut->children();
  int64_t vPreNodeColNums = vPreNodeColNames.size();
  int64_t sCurrentColId = -1;

  VELOX_CHECK_EQ(vPreNodeColNums, vPreNodeColTypes.size());

  for (int64_t i = 0; i < vPreNodeColNums; i++) {
    if (vPreNodeColNames[i] == vExprName && vPreNodeColTypes[i] == vExprType) {
      sCurrentColId = i;
      break;
    }
  }

  if (sCurrentColId == -1) {
    sCurrentColId = vPreNodeColNums + 1;
  }
  sDirectStruct->set_field(sCurrentColId);

  return sFieldExpr;
}

::substrait::Expression* VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const CallTypedExpr>& vCallTypeExpr,
    const RowTypePtr& vPreNodeOutPut) {
  ::substrait::Expression* sExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression>(&arena);

  auto vExprType = vCallTypeExpr->type();
  auto vCallTypeInputs = vCallTypeExpr->inputs();
  std::string vCallTypeExprFunName = vCallTypeExpr->name();

  // different by function names.
  // TODO add support for if Expr and switch Expr
  if (vCallTypeExprFunName != exec::kIf &&
      vCallTypeExprFunName != exec::kSwitch) {
    ::substrait::Expression_ScalarFunction* sScalaExpr =
        sExpr->mutable_scalar_function();

    // TODO need to change yaml file to register function, now is dummy.

    uint32_t sFunId =
        v2SFuncConvertor_.registerSubstraitFunction(vCallTypeExprFunName);
    sScalaExpr->set_function_reference(sFunId);

    for (auto& vArg : vCallTypeInputs) {
      sScalaExpr->add_args()->MergeFrom(
          *toSubstraitExpr(arena, vArg, vPreNodeOutPut));
    }
    ::substrait::Type* sFunType = sScalaExpr->mutable_output_type();
    v2STypeConvertor_.toSubstraitType(arena, vExprType, sFunType);
  } else {
    VELOX_NYI("Unsupport CallTypeExpr with FunName '{}'", vCallTypeExprFunName);
  }

  return sExpr;
}

::substrait::Expression_Literal* VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const ConstantTypedExpr>& vConstExpr,
    const RowTypePtr& vPreNodeOutPut,
    ::substrait::Expression_Literal_Struct* sLitValue) {
  if (vConstExpr->hasValueVector()) {
    return toSubstraitLiteral(arena, vConstExpr->valueVector(), sLitValue);
  } else {
    return toSubstraitLiteral(arena, vConstExpr->value());
  }
}

::substrait::Expression_Literal*
VeloxToSubstraitExprConvertor::toSubstraitLiteral(
    google::protobuf::Arena& arena,
    const velox::variant& vConstExpr) {
  ::substrait::Expression_Literal* sLiteralExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression_Literal>(
          &arena);
  switch (vConstExpr.kind()) {
    case velox::TypeKind::DOUBLE: {
      sLiteralExpr->set_fp64(vConstExpr.value<TypeKind::DOUBLE>());
      break;
    }
    case velox::TypeKind::VARCHAR: {
      ::substrait::Expression_Literal::VarChar* sVarChar =
          google::protobuf::Arena::CreateMessage<
              ::substrait::Expression_Literal::VarChar>(&arena);
      auto vCharValue = vConstExpr.value<StringView>();
      sVarChar->set_value(vCharValue.data());
      sVarChar->set_length(vCharValue.size());
      sLiteralExpr->set_allocated_var_char(sVarChar);
      break;
    }
    case velox::TypeKind::BIGINT: {
      sLiteralExpr->set_i64(vConstExpr.value<TypeKind::BIGINT>());
      break;
    }
    case velox::TypeKind::INTEGER: {
      sLiteralExpr->set_i32(vConstExpr.value<TypeKind::INTEGER>());
      break;
    }
    case velox::TypeKind::SMALLINT: {
      sLiteralExpr->set_i16(vConstExpr.value<TypeKind::INTEGER>());
      break;
    }
    case velox::TypeKind::TINYINT: {
      sLiteralExpr->set_i8(vConstExpr.value<TypeKind::INTEGER>());
      break;
    }
    case velox::TypeKind::BOOLEAN: {
      sLiteralExpr->set_boolean(vConstExpr.value<TypeKind::BOOLEAN>());
      break;
    }
    case velox::TypeKind::REAL: {
      sLiteralExpr->set_fp32(vConstExpr.value<TypeKind::REAL>());
      break;
    }
    case velox::TypeKind::TIMESTAMP: {
      // TODO make sure the type convertor is equal
      sLiteralExpr->set_timestamp(
          vConstExpr.value<TypeKind::TIMESTAMP>().getNanos());
      break;
    }
    default:
      VELOX_NYI(
          "Unsupported constant Type '{}' ",
          mapTypeKindToName(vConstExpr.kind()));
  }

  return sLiteralExpr;
}

::substrait::Expression_Literal*
VeloxToSubstraitExprConvertor::toSubstraitLiteral(
    google::protobuf::Arena& arena,
    const velox::VectorPtr& vVectorValue,
    ::substrait::Expression_Literal_Struct* sLitValue) {
  ::substrait::Expression_Literal* sField =
      google::protobuf::Arena::CreateMessage<::substrait::Expression_Literal>(
          &arena);
  const TypePtr& childType = vVectorValue->type();

  switch (childType->kind()) {
    case velox::TypeKind::BOOLEAN: {
      auto childToFlatVec = vVectorValue->asFlatVector<bool>();
      // get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        sField = sLitValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // process the null value
          sField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          sField->set_boolean(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::TINYINT: {
      auto childToFlatVec = vVectorValue->asFlatVector<int8_t>();
      // get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        sField = sLitValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // process the null value
          sField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          sField->set_i8(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::SMALLINT: {
      auto childToFlatVec = vVectorValue->asFlatVector<int16_t>();
      // get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        sField = sLitValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // process the null value
          sField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          sField->set_i16(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::INTEGER: {
      auto childToFlatVec = vVectorValue->asFlatVector<int32_t>();
      // get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        sField = sLitValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // process the null value
          sField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          sField->set_i32(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::BIGINT: {
      auto childToFlatVec = vVectorValue->asFlatVector<int64_t>();
      // get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        sField = sLitValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // process the null value
          sField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          sField->set_i64(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::REAL: {
      auto childToFlatVec = vVectorValue->asFlatVector<float_t>();
      // get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        sField = sLitValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // process the null value
          sField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          sField->set_fp32(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::DOUBLE: {
      auto childToFlatVec = vVectorValue->asFlatVector<double_t>();
      // get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        sField = sLitValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // process the null value
          sField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          sField->set_fp64(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::VARCHAR: {
      auto childToFlatVec = vVectorValue->asFlatVector<StringView>();
      // get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        sField = sLitValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // process the null value
          sField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          ::substrait::Expression_Literal::VarChar* sVarChar =
              google::protobuf::Arena::CreateMessage<
                  ::substrait::Expression_Literal::VarChar>(&arena);
          StringView vChildValueAt = childToFlatVec->valueAt(i);
          sVarChar->set_value(vChildValueAt);
          sVarChar->set_length(vChildValueAt.size());
          sField->set_allocated_var_char(sVarChar);
        }
      }
      break;
    }
    default: {
      VELOX_UNSUPPORTED(
          "Unsupported type '{}'", std::string(childType->kindName()));
    }
  }
  return sField;
}

::substrait::Expression_Literal*
VeloxToSubstraitExprConvertor::toSubstraitNullLiteral(
    google::protobuf::Arena& arena,
    const velox::TypePtr& vValueType) {
  ::substrait::Expression_Literal* sField =
      google::protobuf::Arena::CreateMessage<::substrait::Expression_Literal>(
          &arena);
  switch (vValueType->kind()) {
    case velox::TypeKind::BOOLEAN: {
      ::substrait::Type_Boolean* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_Boolean>(
              &arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_bool_(nullValue);
      break;
    }
    case velox::TypeKind::TINYINT: {
      ::substrait::Type_I8* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_I8>(&arena);

      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_i8(nullValue);
      break;
    }
    case velox::TypeKind::SMALLINT: {
      ::substrait::Type_I16* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_I16>(&arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_i16(nullValue);
      break;
    }
    case velox::TypeKind::INTEGER: {
      ::substrait::Type_I32* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_I32>(&arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_i32(nullValue);
      break;
    }
    case velox::TypeKind::BIGINT: {
      ::substrait::Type_I64* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_I64>(&arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_i64(nullValue);
      break;
    }
    case velox::TypeKind::VARCHAR: {
      ::substrait::Type_VarChar* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_VarChar>(
              &arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_varchar(nullValue);
      break;
    }
    case velox::TypeKind::REAL: {
      ::substrait::Type_FP32* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_FP32>(
              &arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_fp32(nullValue);
      break;
    }
    case velox::TypeKind::DOUBLE: {
      ::substrait::Type_FP64* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_FP64>(
              &arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      sField->mutable_null()->set_allocated_fp64(nullValue);
      break;
    }
    default: {
      VELOX_UNSUPPORTED(
          "Unsupported type '{}'", std::string(vValueType->kindName()));
    }
  }
  return sField;
}

} // namespace facebook::velox::substrait
