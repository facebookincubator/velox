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
    const std::shared_ptr<const ITypedExpr>& expr,
    const RowTypePtr& inputType) {
  ::substrait::Expression* substraitExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression>(&arena);
  if (auto constExpr =
          std::dynamic_pointer_cast<const ConstantTypedExpr>(expr)) {
    substraitExpr->set_allocated_literal(toSubstraitExpr(arena, constExpr));
    return substraitExpr;
  }
  if (auto callTypeExpr =
          std::dynamic_pointer_cast<const CallTypedExpr>(expr)) {
    substraitExpr->MergeFrom(*toSubstraitExpr(arena, callTypeExpr, inputType));
    return substraitExpr;
  }
  if (auto fieldExpr =
          std::dynamic_pointer_cast<const FieldAccessTypedExpr>(expr)) {
    substraitExpr->set_allocated_selection(
        toSubstraitExpr(arena, fieldExpr, inputType));
    return substraitExpr;
  }
  if (auto castExpr = std::dynamic_pointer_cast<const CastTypedExpr>(expr)) {
    substraitExpr->set_allocated_cast(
        toSubstraitExpr(arena, castExpr, inputType));
    return substraitExpr;
  } else {
    VELOX_UNSUPPORTED("Unsupport Expr '{}' in Substrait", expr->toString());
  }
}

::substrait::Expression_Cast* VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const CastTypedExpr>& castExpr,
    const RowTypePtr& inputType) {
  ::substrait::Expression_Cast* substraitCastExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression_Cast>(
          &arena);
  std::vector<std::shared_ptr<const ITypedExpr>> castExprInputs =
      castExpr->inputs();

  substraitCastExpr->set_allocated_type(
      typeConvertor_.toSubstraitType(arena, castExpr->type()));

  for (auto& arg : castExprInputs) {
    substraitCastExpr->set_allocated_input(
        toSubstraitExpr(arena, arg, inputType));
  }
  return substraitCastExpr;
}

::substrait::Expression_FieldReference*
VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const FieldAccessTypedExpr>& fieldExpr,
    const RowTypePtr& inputType) {
  ::substrait::Expression_FieldReference* substraitFieldExpr =
      google::protobuf::Arena::CreateMessage<
          ::substrait::Expression_FieldReference>(&arena);

  const auto& exprType = fieldExpr->type();
  std::string exprName = fieldExpr->name();

  ::substrait::Expression_ReferenceSegment_StructField* directStruct =
      substraitFieldExpr->mutable_direct_reference()->mutable_struct_field();

  const auto& preNodeColNames = inputType->names();
  const auto& preNodeColTypes = inputType->children();
  const auto& preNodeColNums = preNodeColNames.size();
  auto currentColId = -1;

  VELOX_CHECK_EQ(preNodeColNums, preNodeColTypes.size());

  for (auto i = 0; i < preNodeColNums; i++) {
    if (preNodeColNames[i] == exprName && preNodeColTypes[i] == exprType) {
      currentColId = i;
      break;
    }
  }

  if (currentColId == -1) {
    currentColId = preNodeColNums + 1;
  }
  directStruct->set_field(currentColId);

  return substraitFieldExpr;
}

::substrait::Expression* VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const CallTypedExpr>& callTypeExpr,
    const RowTypePtr& inputType) {
  ::substrait::Expression* substraitExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression>(&arena);

  auto callTypeExprInputs = callTypeExpr->inputs();
  std::string callTypeExprFunName = callTypeExpr->name();

  // The processing is different for different function names.
  // TODO add support for if Expr and switch Expr
  if (callTypeExprFunName != exec::kIf &&
      callTypeExprFunName != exec::kSwitch) {
    ::substrait::Expression_ScalarFunction* scalaExpr =
        substraitExpr->mutable_scalar_function();

    // TODO need to change yaml file to register function, now is dummy.

    scalaExpr->set_function_reference(
        funcConvertor_.registerSubstraitFunction(callTypeExprFunName));

    for (auto& arg : callTypeExprInputs) {
      scalaExpr->add_args()->MergeFrom(*toSubstraitExpr(arena, arg, inputType));
    }

    scalaExpr->set_allocated_output_type(
        typeConvertor_.toSubstraitType(arena, callTypeExpr->type()));
  } else {
    VELOX_NYI("Unsupport CallTypeExpr with FunName '{}'", callTypeExprFunName);
  }

  return substraitExpr;
}

::substrait::Expression_Literal* VeloxToSubstraitExprConvertor::toSubstraitExpr(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const ConstantTypedExpr>& constExpr,
    ::substrait::Expression_Literal_Struct* litValue) {
  if (constExpr->hasValueVector()) {
    return toSubstraitLiteral(arena, constExpr->valueVector(), litValue);
  } else {
    return toSubstraitLiteral(arena, constExpr->value());
  }
}

::substrait::Expression_Literal*
VeloxToSubstraitExprConvertor::toSubstraitLiteral(
    google::protobuf::Arena& arena,
    const velox::variant& variantValue) {
  ::substrait::Expression_Literal* literalExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression_Literal>(
          &arena);
  switch (variantValue.kind()) {
    case velox::TypeKind::DOUBLE: {
      literalExpr->set_fp64(variantValue.value<TypeKind::DOUBLE>());
      break;
    }
    case velox::TypeKind::VARCHAR: {
      ::substrait::Expression_Literal::VarChar* substraitVarChar =
          google::protobuf::Arena::CreateMessage<
              ::substrait::Expression_Literal::VarChar>(&arena);
      auto charValue = variantValue.value<StringView>();
      substraitVarChar->set_value(charValue.data());
      substraitVarChar->set_length(charValue.size());
      literalExpr->set_allocated_var_char(substraitVarChar);
      break;
    }
    case velox::TypeKind::BIGINT: {
      literalExpr->set_i64(variantValue.value<TypeKind::BIGINT>());
      break;
    }
    case velox::TypeKind::INTEGER: {
      literalExpr->set_i32(variantValue.value<TypeKind::INTEGER>());
      break;
    }
    case velox::TypeKind::SMALLINT: {
      literalExpr->set_i16(variantValue.value<TypeKind::INTEGER>());
      break;
    }
    case velox::TypeKind::TINYINT: {
      literalExpr->set_i8(variantValue.value<TypeKind::INTEGER>());
      break;
    }
    case velox::TypeKind::BOOLEAN: {
      literalExpr->set_boolean(variantValue.value<TypeKind::BOOLEAN>());
      break;
    }
    case velox::TypeKind::REAL: {
      literalExpr->set_fp32(variantValue.value<TypeKind::REAL>());
      break;
    }
    case velox::TypeKind::TIMESTAMP: {
      // TODO make sure the type convertor is equal
      literalExpr->set_timestamp(
          variantValue.value<TypeKind::TIMESTAMP>().getNanos());
      break;
    }
    default:
      VELOX_NYI(
          "Unsupported constant Type '{}' ",
          mapTypeKindToName(variantValue.kind()));
  }

  return literalExpr;
}

::substrait::Expression_Literal*
VeloxToSubstraitExprConvertor::toSubstraitLiteral(
    google::protobuf::Arena& arena,
    const velox::VectorPtr& vectorValue,
    ::substrait::Expression_Literal_Struct* litValue) {
  ::substrait::Expression_Literal* substraitField =
      google::protobuf::Arena::CreateMessage<::substrait::Expression_Literal>(
          &arena);
  const TypePtr& childType = vectorValue->type();

  switch (childType->kind()) {
    case velox::TypeKind::BOOLEAN: {
      auto childToFlatVec = vectorValue->asFlatVector<bool>();
      // Get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        substraitField = litValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // Process the null value.
          substraitField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          substraitField->set_boolean(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::TINYINT: {
      auto childToFlatVec = vectorValue->asFlatVector<int8_t>();
      // Get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        substraitField = litValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // Process the null value.
          substraitField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          substraitField->set_i8(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::SMALLINT: {
      auto childToFlatVec = vectorValue->asFlatVector<int16_t>();
      // Get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        substraitField = litValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // Process the null value.
          substraitField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          substraitField->set_i16(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::INTEGER: {
      auto childToFlatVec = vectorValue->asFlatVector<int32_t>();
      // Get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        substraitField = litValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // Process the null value.
          substraitField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          substraitField->set_i32(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::BIGINT: {
      auto childToFlatVec = vectorValue->asFlatVector<int64_t>();
      // Get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        substraitField = litValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // Process the null value.
          substraitField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          substraitField->set_i64(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::REAL: {
      auto childToFlatVec = vectorValue->asFlatVector<float_t>();
      // Get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        substraitField = litValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // Process the null value.
          substraitField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          substraitField->set_fp32(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::DOUBLE: {
      auto childToFlatVec = vectorValue->asFlatVector<double_t>();
      // Get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        substraitField = litValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // Process the null value.
          substraitField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          substraitField->set_fp64(childToFlatVec->valueAt(i));
        }
      }
      break;
    }
    case velox::TypeKind::VARCHAR: {
      auto childToFlatVec = vectorValue->asFlatVector<StringView>();
      // Get the batchSize and convert each value in it.
      vector_size_t flatVecSize = childToFlatVec->size();
      for (int64_t i = 0; i < flatVecSize; i++) {
        substraitField = litValue->add_fields();
        if (childToFlatVec->isNullAt(i)) {
          // Process the null value.
          substraitField->MergeFrom(*toSubstraitNullLiteral(arena, childType));
        } else {
          ::substrait::Expression_Literal::VarChar* sVarChar =
              google::protobuf::Arena::CreateMessage<
                  ::substrait::Expression_Literal::VarChar>(&arena);
          StringView vChildValueAt = childToFlatVec->valueAt(i);
          sVarChar->set_value(vChildValueAt);
          sVarChar->set_length(vChildValueAt.size());
          substraitField->set_allocated_var_char(sVarChar);
        }
      }
      break;
    }
    default: {
      VELOX_UNSUPPORTED("Unsupported type '{}'", childType->toString());
    }
  }
  return substraitField;
}

::substrait::Expression_Literal*
VeloxToSubstraitExprConvertor::toSubstraitNullLiteral(
    google::protobuf::Arena& arena,
    const velox::TypePtr& type) {
  ::substrait::Expression_Literal* substraitField =
      google::protobuf::Arena::CreateMessage<::substrait::Expression_Literal>(
          &arena);
  switch (type->kind()) {
    case velox::TypeKind::BOOLEAN: {
      ::substrait::Type_Boolean* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_Boolean>(
              &arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      substraitField->mutable_null()->set_allocated_bool_(nullValue);
      break;
    }
    case velox::TypeKind::TINYINT: {
      ::substrait::Type_I8* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_I8>(&arena);

      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      substraitField->mutable_null()->set_allocated_i8(nullValue);
      break;
    }
    case velox::TypeKind::SMALLINT: {
      ::substrait::Type_I16* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_I16>(&arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      substraitField->mutable_null()->set_allocated_i16(nullValue);
      break;
    }
    case velox::TypeKind::INTEGER: {
      ::substrait::Type_I32* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_I32>(&arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      substraitField->mutable_null()->set_allocated_i32(nullValue);
      break;
    }
    case velox::TypeKind::BIGINT: {
      ::substrait::Type_I64* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_I64>(&arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      substraitField->mutable_null()->set_allocated_i64(nullValue);
      break;
    }
    case velox::TypeKind::VARCHAR: {
      ::substrait::Type_VarChar* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_VarChar>(
              &arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      substraitField->mutable_null()->set_allocated_varchar(nullValue);
      break;
    }
    case velox::TypeKind::REAL: {
      ::substrait::Type_FP32* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_FP32>(
              &arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      substraitField->mutable_null()->set_allocated_fp32(nullValue);
      break;
    }
    case velox::TypeKind::DOUBLE: {
      ::substrait::Type_FP64* nullValue =
          google::protobuf::Arena::CreateMessage<::substrait::Type_FP64>(
              &arena);
      nullValue->set_nullability(
          ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
      substraitField->mutable_null()->set_allocated_fp64(nullValue);
      break;
    }
    default: {
      VELOX_UNSUPPORTED("Unsupported type '{}'", std::string(type->kindName()));
    }
  }
  return substraitField;
}

} // namespace facebook::velox::substrait
