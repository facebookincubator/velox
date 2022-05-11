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

#include <folly/container/F14Map.h>

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
template <typename T>
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

template <typename T>
void ArraySumFunction<T>::apply(
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
  TypePtr type = arrayVector->type()->childAt(0);
  auto resultVector = BaseVector::create(type, numRows, pool);

  // Get access to raw values for the result
  T* resultValues = (T*) resultVector->valuesAsVoid();

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

      T sum = 0;
      for (; start < end; start++) {
        if (!elements->isNullAt(start)) {
          sum += elements->template valueAt<T>(start);
        }
      }

      // Set the value at i equal to the sum
      resultValues[i] = sum;
    }
  }

  context->moveOrCopyResult(resultVector, rows, result);
}

template <>
void ArraySumFunction<Timestamp>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {}

template <>
void ArraySumFunction<StringView>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
    const TypePtr& outputType,
    exec::EvalCtx* context,
    VectorPtr* result) const {}

template <>
void ArraySumFunction<Date>::apply(
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

 using T = typename TypeTraits<kind>::NativeType;
 return std::make_shared<ArraySumFunction<T>>();
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
         .returnType("array(tinyint)")
         .argumentType("bigint")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("array(smallint)")
         .argumentType("bigint")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("array(integer)")
         .argumentType("bigint")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("array(bigint)")
         .argumentType("bigint")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("array(real)")
         .argumentType("double")
         .build(),
     exec::FunctionSignatureBuilder()
         .returnType("array(double)")
         .argumentType("double")
         .build()};
}

} // namespace

// Register function.
VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
   udf_array_sum,
   signatures(),
   create);

} // namespace facebook::velox::functions

