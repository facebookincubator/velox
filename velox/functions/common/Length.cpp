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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/StringEncodingUtils.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook {
namespace velox {
namespace functions {

namespace {
/**
 * Length of the input UTF8 string in characters.
 * Have an Ascii fast path optimization
 **/
template <typename T>
class LengthFunction : public exec::VectorFunction {
 private:
  // String encoding wrappable function
  template <StringEncodingMode stringEncoding>
  struct ApplyInternalString {
    static void apply(
        const SelectivityVector& rows,
        const DecodedVector* decodedInput,
        FlatVector<T>* resultFlatVector) {
      rows.applyToSelected([&](int row) {
        auto result = stringImpl::length<stringEncoding>(
            decodedInput->valueAt<StringView>(row));
        resultFlatVector->set(row, result);
      });
    }
  };

 public:
  bool isDefaultNullBehavior() const override {
    return true;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      exec::Expr* /* unused */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    auto inputArg = args.at(0);
    // Decode input argument
    exec::LocalDecodedVector inputHolder(context, *inputArg, rows);
    auto decodedInput = inputHolder.get();

    // Prepare output vector
    BaseVector::ensureWritable(
        rows, CppToType<T>::ImplType::create(), context->pool(), result);
    auto* resultFlatVector = (*result)->as<FlatVector<T>>();

    if (inputArg->typeKind() == TypeKind::VARCHAR) {
      auto stringEncoding = getStringEncodingOrUTF8(inputArg.get());
      StringEncodingTemplateWrapper<ApplyInternalString>::apply(
          stringEncoding, rows, decodedInput, resultFlatVector);
      return;
    } else if (inputArg->typeKind() == TypeKind::VARBINARY) {
      rows.applyToSelected([&](int row) {
        resultFlatVector->set(
            row, decodedInput->valueAt<StringView>(row).size());
      });
      return;
    }
    VELOX_UNREACHABLE();
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // varchar -> bigint
    return {
        exec::FunctionSignatureBuilder()
            .returnType(CppToType<T>::name)
            .argumentType("varchar")
            .build(),
        exec::FunctionSignatureBuilder()
            .returnType(CppToType<T>::name)
            .argumentType("varbinary")
            .build(),
    };
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_length,
    LengthFunction<int64_t>::signatures(),
    std::make_unique<LengthFunction<int64_t>>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_length_int32,
    LengthFunction<int32_t>::signatures(),
    std::make_unique<LengthFunction<int32_t>>());

} // namespace functions
} // namespace velox
} // namespace facebook
