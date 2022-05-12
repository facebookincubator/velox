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

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/ComparatorUtil.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"

namespace facebook::velox::functions {
namespace {

// See documentation at https://prestodb.io/docs/current/functions/array.html
///
/// Implements the array_sum function.
///
template <typename IT, typename OT>
class ArraySumFunction : public exec::VectorFunction {
public:
 // Execute function.
 void apply(
     const SelectivityVector& rows,
     std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
     const TypePtr& outputType,
     exec::EvalCtx* context,
     VectorPtr* result) const override;
};

template <typename IT, typename OT>
void ArraySumFunction<IT, OT>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {
  // Acquire the array elements vector.
  auto arrayVector = args[0]->as<ArrayVector>();
  VELOX_CHECK(arrayVector);
  auto elementsVector = arrayVector->elements();

  auto elementsRows =
      toElementRows(elementsVector->size(), rows, arrayVector);
  exec::LocalDecodedVector elements(context, *elementsVector, elementsRows);

  vector_size_t numRows = arrayVector->size();

  // Allocate new vector for the result
  memory::MemoryPool* pool = context->pool();
  auto resultVector = BaseVector::create(outputType, numRows, pool);


  // Get access to raw values for the result
  OT* resultValues = (OT*) resultVector->valuesAsVoid();

  // Iterate over the input vector and find the sum of each array's values
  for (int i = 0; i < numRows; i++) {
    // If the whole array is null then set the row null in the output
    if (arrayVector->isNullAt(i)) {
      resultVector->setNull(i, true);
    }
    // If the array is not null then sum the elements and set the result to the sum
    else {
      int start = arrayVector->offsetAt(i);
      int end = start + arrayVector->sizeAt(i);

      OT sum = 0;
      for (; start < end; start++) {
        if (!elements->isNullAt(start)) {
          sum += elements->template valueAt<IT>(start);
        }
      }

      // Set the value at i equal to the sum
      resultValues[i] = sum;
    }
  }

  context->moveOrCopyResult(resultVector, rows, result);
}

template <>
void ArraySumFunction<Timestamp, int64_t>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {}

template <>
void ArraySumFunction<Timestamp, double>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {}

template <>
void ArraySumFunction<StringView, int64_t>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {}

template <>
void ArraySumFunction<StringView, double>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {}

template <>
void ArraySumFunction<Date, int64_t>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {}

template <>
void ArraySumFunction<Date, double>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {}

// Validate number of parameters and types.
void validateType(const std::vector<exec::VectorFunctionArg>& inputArgs) {
 VELOX_USER_CHECK_EQ(
     inputArgs.size(), 1, "array_sum requires exactly one parameter");

 auto arrayType = inputArgs.front().type;
 VELOX_USER_CHECK_EQ(
     arrayType->kind(),
     TypeKind::ARRAY,
     "array_sum requires argument of type ARRAY");
}

// Create function template based on type.
template <TypeKind kind>
std::shared_ptr<exec::VectorFunction> createTyped(
   const std::vector<exec::VectorFunctionArg>& inputArgs) {
 VELOX_CHECK_EQ(inputArgs.size(), 1);

 using IT = typename TypeTraits<kind>::NativeType;
 if (kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT
     || kind == TypeKind::INTEGER || kind == TypeKind::BIGINT) {
   return std::make_shared<ArraySumFunction<IT, int64_t>>();
 }
 if (kind == TypeKind::REAL || kind == TypeKind::DOUBLE) {
   return std::make_shared<ArraySumFunction<IT, double>>();
 }
 VELOX_FAIL()
}

// Create function.
std::shared_ptr<exec::VectorFunction> create(
   const std::string& /* name */,
   const std::vector<exec::VectorFunctionArg>& inputArgs) {
 validateType(inputArgs);
 auto elementType = inputArgs.front().type->childAt(0);

 return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
     createTyped, elementType->kind(), inputArgs);
}

// Define function signature.
// array(T1) -> T2 where T must be coercible to bigint or double, and
// T2 is bigint or double
std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
 return {
     exec::FunctionSignatureBuilder()
         .returnType("bigint")
         .argumentType("array(tinyint)")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("bigint")
         .argumentType("array(smallint)")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("bigint")
         .argumentType("array(integer)")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("bigint")
         .argumentType("array(bigint)")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("double")
         .argumentType("array(real)")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("double")
         .argumentType("array(double)")
         .build()};
}

} // namespace

// Register function.
VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
   udf_array_sum,
   signatures(),
   create);

} // namespace facebook::velox::functions

