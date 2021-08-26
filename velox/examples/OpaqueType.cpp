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

#include "velox/common/memory/Memory.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

using namespace facebook::velox;

VectorPtr evaluate(
    const std::string& functionName,
    const std::string& argName1,
    const std::string& argName2,
    core::ExecCtx& execCtx,
    RowVectorPtr rowVector);

/// This file exemplifies how OPAQUE types are used in vectors, simple and
/// vectorized functions, and the expression eval engine.
///
/// First, two functions that operate over opaque types and implement the same
/// logic are defined and registered, one simple and one vectorized. Then, we
/// show how opaque objects can be wrapped in Velox vectors, and how they can be
/// pushed through the expression evaluation engine until they are available to
/// the user-defined function code.

// We first define two classes to be carried by the opaque type container.
// In real scenarios, these classes are usually used to hold large state buffers
// (like ML models), so we disable the copy constructor, just in case:
struct OpaqueInput {
  // Let's also count the number of instances, just to ensure no inadvertent
  // copies of this potentially large object will be made.
  OpaqueInput() {
    ++OpaqueInput::numInstances;
  }

  // No copies.
  OpaqueInput(OpaqueInput const&) = delete;
  OpaqueInput& operator=(OpaqueInput const&) = delete;

  // In this toy example, this class holds a map between integers and strings.
  std::string get(size_t idx) const {
    auto it = map_.find(idx);
    if (it != map_.end()) {
      return it->second;
    }
    return "";
  }

  // This could be the map that holds the large state.
  std::unordered_map<size_t, std::string> map_{
      {0, "zero"},
      {1, "one"},
      {2, "two"},
      {3, "three"},
      {4, "four"},
  };

  static size_t numInstances;
};
size_t OpaqueInput::numInstances{0};

// Our second opaqued class will be just an std::pair. Make sure it's printable.
using OpaqueOutput = std::pair<size_t, std::string>;
inline std::ostream& operator<<(std::ostream& out, const OpaqueOutput& opaque) {
  return out << "[" << opaque.first << ": " << opaque.second << "]";
}

// Now we define a simple function that takes an OpaqueInput (via opaque type),
// and returns an OpaqueOutput type.
//
// This function takes two parameters, an OpaqueInput (via opaque type), which
// contains the large map defined above, and a bigint. The function looks up the
// bigint inside OpaqueInput to find a corresponding string, and returns an
// OpaqueOutput, which holds a pair containing the idx and string. It doesn't
// implement any interesting logic, but exemplifies how to take an opaque object
// in, and return another.
//
// Check `velox/docs/develop/scalar-functions.rst` for more documentation on how
// to build scalar functions.
VELOX_UDF_BEGIN(opaque_simple)
FOLLY_ALWAYS_INLINE bool call(
    arg_type<std::shared_ptr<OpaqueOutput>>& out,
    const arg_type<std::shared_ptr<OpaqueInput>>& state,
    const int64_t& idx) {
  out = std::make_shared<OpaqueOutput>(idx, state->get(idx));
  return true;
}
VELOX_UDF_END()

// Next, we implement the same logic using the vectorized function framework.
class OpaqueVectorFunction : public exec::VectorFunction {
 public:
  // Note that when using the vectorized function framework, you will need to
  // explicitly cast your opaque type vector into the expected opaqued type.
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      exec::Expr* /*caller*/,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    // First capture the input pointers.
    VELOX_CHECK_EQ(2, args.size());
    BaseVector* opaqueInput = args[0].get();
    BaseVector* flatInput = args[1].get();

    // We cannot make any assumptions about the encoding of the input vector. In
    // order to access their data as FlatVectors, we need to decode them first:
    exec::LocalDecodedVector opaqueHolder(context, *opaqueInput, rows);
    auto decodedInput = opaqueHolder.get();

    // Decode second argument.
    exec::LocalDecodedVector bigintHolder(context, *flatInput, rows);
    auto decodedBigint = bigintHolder.get();

    // Ensure we have an output vector where we can write the output opaqued
    // values.
    BaseVector::ensureWritable(
        rows, OPAQUE<OpaqueOutput>(), context->pool(), result);
    auto* output = (*result)->as<KindToFlatVector<TypeKind::OPAQUE>::type>();

    // `applyToSelected()` will execute the lambda below on each row enabled in
    // the input selectivity vector (rows):
    rows.applyToSelected([&](vector_size_t row) {
      // FlatVectors of opaque types use std::shared_ptr<void> by default. Cast
      // it to our opaque type (OpaqueInput).
      auto opaqueInput =
          decodedInput->valueAt<std::shared_ptr<OpaqueInput>>(row);

      // Allocate a new OpaqueOutput and save a shared_ptr to it in the output
      // vector.
      output->set(
          row, std::make_shared<OpaqueOutput>(row, opaqueInput->get(row)));
    });
  }

  // Define the valid function signatures. We currently don't support type
  // matching in the opaqued type (OpaqueInput or OpaqueOutput) when using the
  // vectorized function framework.
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("opaque")
                .argumentType("opaque")
                .argumentType("bigint")
                .build()};
  }
};

// Declaring the vectorized function.
VELOX_DECLARE_VECTOR_FUNCTION(
    udf_opaque_vector,
    OpaqueVectorFunction::signatures(),
    std::make_unique<OpaqueVectorFunction>());

int main(int argc, char** argv) {
  // Registering both simple and vectorized functions.
  registerFunction<
      udf_opaque_simple,
      std::shared_ptr<OpaqueOutput>,
      std::shared_ptr<OpaqueInput>,
      int64_t>();
  VELOX_REGISTER_VECTOR_FUNCTION(udf_opaque_vector, "opaque_vector");

  // Create memory pool and other query-related structures.
  auto queryCtx = core::QueryCtx::create();
  auto pool = memory::getDefaultScopedMemoryPool();
  core::ExecCtx execCtx{std::move(pool), queryCtx.get()};

  // Next, we need to generate an input batch of data (rowVector). We create a
  // small batch of `vectorSize` rows, and three columns:
  //
  // 1. A FlatVector of Opaque<OpaqueInput>. Since we don't want to hold
  // multiple copies of OpaqueInput in memory (remember this can be large), we
  // add one shared_ptr to the single object per row.
  //
  // 2. A ConstantVector of Opaque<OpaqueInput>. This is similar to #1, but this
  // is smarter and encodes the opaque type as a constant (since it's
  // essentially a repetition of the same value).
  //
  // 3. A regular FlatVector<BIGINT> with monotonically increasing numbers
  // starting in zero.
  const size_t vectorSize = 5;
  using FlatVectorOpaque = KindToFlatVector<TypeKind::OPAQUE>::type;
  auto inputRowType = ROW({
      {"flat_opaque", OPAQUE<OpaqueInput>()},
      {"constant_opaque", OPAQUE<OpaqueInput>()},
      {"bigint", BIGINT()},
  });

  // Create vector #1:
  auto vector1 =
      BaseVector::create(inputRowType->childAt(0), vectorSize, execCtx.pool());
  auto opaqueVector = vector1->as<FlatVectorOpaque>();

  // Create the single instance of OpaqueInput, and add multiple shared_ptr to
  // it in the flatVector.
  auto opaqueObj = std::make_shared<OpaqueInput>();
  for (size_t i = 0; i < opaqueVector->size(); i++) {
    opaqueVector->set(i, opaqueObj);
  }

  // Create vector #2. Just a constant to the shared_ptr we created above.
  auto vector2 = BaseVector::createConstant(
      variant::opaque(opaqueObj), vectorSize, execCtx.pool());

  // Create vector #3. The monotinically increasing flatVector<bigint>.
  auto vector3 = std::dynamic_pointer_cast<FlatVector<int64_t>>(
      BaseVector::create(BIGINT(), vectorSize, execCtx.pool()));
  auto rawValues = vector3->mutableRawValues();
  std::iota(rawValues, rawValues + vectorSize, 0); // 0, 1, 2, 3, ...

  // Wrap the three vectors above in a RowVector (the input batch).
  auto rowVector = std::make_shared<RowVector>(
      execCtx.pool(),
      inputRowType,
      BufferPtr(nullptr),
      vectorSize,
      std::vector<VectorPtr>{vector1, vector2, vector3});

  // Now, let's go ahead and first execute the simple function.
  //
  // The helper `evaluate()` function basically creates and executes the
  // following expression:
  //
  //   opaque_simple(flat_opaque, bigint);
  //
  LOG(INFO) << "Executing simple opaque function:";
  auto outputVector = std::dynamic_pointer_cast<FlatVectorOpaque>(
      evaluate("opaque_simple", "flat_opaque", "bigint", execCtx, rowVector));

  // Print the output values so we can verify them, just for fun.
  for (size_t i = 0; i < outputVector->size(); ++i) {
    auto opaque =
        std::static_pointer_cast<OpaqueOutput>(outputVector->valueAt(i));
    LOG(INFO) << "Found OpaqueOutput: " << *opaque;
  }

  // Same thing, but now using the vectorized function ("opaque_vector").
  LOG(INFO) << "Executing vectorized opaque function:";
  outputVector = std::dynamic_pointer_cast<FlatVectorOpaque>(
      evaluate("opaque_vector", "flat_opaque", "bigint", execCtx, rowVector));

  // Print it again.
  for (size_t i = 0; i < outputVector->size(); ++i) {
    auto opaque =
        std::static_pointer_cast<OpaqueOutput>(outputVector->valueAt(i));
    LOG(INFO) << "Found OpaqueOutput: " << *opaque;
  }

  // Lastly, let's execute the same vectorized function, but now over the opaque
  // column encoded as constant (constant_opaque), just to ensure it returns the
  // same result.
  LOG(INFO) << "Executing vectorized opaque function "
               "(over constant opaque col):";
  outputVector = std::dynamic_pointer_cast<FlatVectorOpaque>(evaluate(
      "opaque_vector", "constant_opaque", "bigint", execCtx, rowVector));

  // Print it.
  for (size_t i = 0; i < outputVector->size(); ++i) {
    auto opaque =
        std::static_pointer_cast<OpaqueOutput>(outputVector->valueAt(i));
    LOG(INFO) << "Found OpaqueOutput: " << *opaque;
  }

  // Verify that no inadvertent copies/instances of the large class were
  // created.
  LOG(INFO) << "Number of instances of OpaqueState: "
            << OpaqueInput::numInstances;
  return 0;
}

// Helper function that creates a simple expression plan and executes it against
// rowVector.
//
// Don't spend too much time trying to follow this code, If you would like a
// more detailed description of this process, please check
// `velox/examples/ExpressionEval.cpp` instead.
VectorPtr evaluate(
    const std::string& functionName,
    const std::string& argName1,
    const std::string& argName2,
    core::ExecCtx& execCtx,
    RowVectorPtr rowVector) {
  std::vector<VectorPtr> result{nullptr};
  SelectivityVector rows{rowVector->size()};

  auto rowType = rowVector->type()->as<TypeKind::ROW>();

  auto inputExprNode =
      std::make_shared<const core::InputTypedExpr>(rowVector->type());
  auto fieldAccessExprNode1 = std::make_shared<core::FieldAccessTypedExpr>(
      rowType.findChild(argName1),
      std::vector<core::TypedExprPtr>{inputExprNode},
      argName1);
  auto fieldAccessExprNode2 = std::make_shared<core::FieldAccessTypedExpr>(
      rowType.findChild(argName2),
      std::vector<core::TypedExprPtr>{inputExprNode},
      argName2);

  auto exprPlan = std::make_shared<core::CallTypedExpr>(
      OPAQUE<OpaqueOutput>(),
      std::vector<core::TypedExprPtr>{
          fieldAccessExprNode1, fieldAccessExprNode2},
      functionName);

  exec::ExprSet exprSet({exprPlan}, &execCtx);
  exec::EvalCtx evalCtx(&execCtx, &exprSet, rowVector.get());
  exprSet.eval(rows, &evalCtx, &result);
  return result.front();
}
