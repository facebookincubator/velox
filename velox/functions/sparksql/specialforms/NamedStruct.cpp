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

#include "velox/functions/sparksql/specialforms/NamedStruct.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

/// VectorFunction implementation for named_struct that creates a RowVector
/// with specified field names and values.
class NamedStructFunction : public exec::VectorFunction {
 public:
  /// @param rowType The result ROW type with field names and types
  /// @param valueIndices Indices of value arguments in the args vector
  ///        (skipping the name arguments which are at even positions)
  NamedStructFunction(
      const TypePtr& rowType,
      const std::vector<int32_t>& valueIndices)
      : rowType_(rowType), valueIndices_(valueIndices) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // Create output RowVector
    auto rowVector = std::dynamic_pointer_cast<RowVector>(
        BaseVector::create(outputType, rows.end(), context.pool()));

    // Set each field from the corresponding value argument
    for (size_t i = 0; i < valueIndices_.size(); ++i) {
      rowVector->childAt(i) = args[valueIndices_[i]];
    }

    context.moveOrCopyResult(rowVector, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // named_struct accepts variadic arguments, but we can't express
    // the alternating name-value pattern in signatures.
    // Type resolution is done in resolveType().
    return {exec::FunctionSignatureBuilder()
                .returnType("row()")
                .argumentType("varchar")
                .argumentType("any")
                .variableArity()
                .build()};
  }

 private:
  const TypePtr rowType_;
  const std::vector<int32_t> valueIndices_;
};

} // namespace

TypePtr NamedStructCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  // Validate we have at least 2 arguments and even number
  VELOX_USER_CHECK_GE(
      argTypes.size(),
      2,
      "named_struct requires at least 2 arguments (1 field)");
  
  VELOX_USER_CHECK_EQ(
      argTypes.size() % 2,
      0,
      "named_struct requires an even number of arguments (name-value pairs)");

  // Build field names and types
  std::vector<std::string> fieldNames;
  std::vector<TypePtr> fieldTypes;

  for (size_t i = 0; i < argTypes.size(); i += 2) {
    // Field name must be VARCHAR
    VELOX_USER_CHECK_EQ(
        argTypes[i]->kind(),
        TypeKind::VARCHAR,
        "Field name at position {} must be VARCHAR, got {}",
        i,
        argTypes[i]->toString());

    // We can't extract the actual string value here (that's done in
    // constructSpecialForm where we have access to ConstantExpr),
    // so we use placeholder names for now. They will be replaced in
    // constructSpecialForm.
    fieldNames.push_back(fmt::format("_field_{}", i / 2));
    fieldTypes.push_back(argTypes[i + 1]);
  }

  return ROW(std::move(fieldNames), std::move(fieldTypes));
}

exec::ExprPtr NamedStructCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK_GE(
      args.size(), 2, "named_struct requires at least 2 arguments");
  VELOX_USER_CHECK_EQ(
      args.size() % 2,
      0,
      "named_struct requires an even number of arguments");

  // Extract field names and validate they are constant expressions
  std::vector<std::string> fieldNames;
  std::vector<int32_t> valueIndices;

  for (size_t i = 0; i < args.size(); i += 2) {
    // Field name must be a constant expression
    auto constantExpr = std::dynamic_pointer_cast<exec::ConstantExpr>(args[i]);
    VELOX_USER_CHECK_NOT_NULL(
        constantExpr,
        "Field name at position {} must be a constant expression",
        i);

    VELOX_USER_CHECK(
        constantExpr->value()->isConstantEncoding(),
        "Field name at position {} must be a constant value",
        i);

    auto constantVector =
        constantExpr->value()->asUnchecked<ConstantVector<StringView>>();
    
    VELOX_USER_CHECK(
        !constantVector->isNullAt(0),
        "Field name at position {} cannot be null",
        i);

    auto fieldName = constantVector->valueAt(0).str();
    fieldNames.push_back(fieldName);
    valueIndices.push_back(i + 1); // Value is at next position
  }

  // Build the actual ROW type with real field names
  std::vector<TypePtr> fieldTypes;
  for (size_t i = 1; i < args.size(); i += 2) {
    fieldTypes.push_back(args[i]->type());
  }

  auto rowType = ROW(std::move(fieldNames), std::move(fieldTypes));

  // Create the function
  auto namedStructFunction =
      std::make_shared<NamedStructFunction>(rowType, valueIndices);

  // Build the expression with only value arguments (names are constants)
  std::vector<exec::ExprPtr> valueArgs;
  for (auto idx : valueIndices) {
    valueArgs.push_back(std::move(args[idx]));
  }

  return std::make_shared<exec::Expr>(
      rowType,
      std::move(valueArgs),
      namedStructFunction,
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
      kNamedStruct,
      trackCpuUsage);
}

} // namespace facebook::velox::functions::sparksql
