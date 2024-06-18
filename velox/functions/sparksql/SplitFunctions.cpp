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

#include <utility>

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/lib/Re2Functions.h"

namespace facebook::velox::functions::sparksql {
namespace {
class Split final : public exec::VectorFunction {
 public:
  Split() {}
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // Get the decoded vectors out of arguments.
    const bool noLimit = (args.size() == 2);
    exec::DecodedArgs decodedArgs(rows, args, context);
    DecodedVector* strings = decodedArgs.at(0);
    DecodedVector* delims = decodedArgs.at(1);
    DecodedVector* limits = noLimit ? nullptr : decodedArgs.at(2);
    BaseVector::ensureWritable(rows, ARRAY(VARCHAR()), context.pool(), result);
    exec::VectorWriter<Array<Varchar>> resultWriter;
    resultWriter.init(*result->as<ArrayVector>());
    int64_t limit = std::numeric_limits<int64_t>::max();
    if (!noLimit) {
      limit = limits->valueAt<int64_t>(0);
      if (limit <= 0) {
        limit = std::numeric_limits<int64_t>::max();
      }
    }
    // Optimization for the (flat, const, const) case.
    if (strings->isIdentityMapping() and delims->isConstantMapping() and
        (noLimit or limits->isConstantMapping())) {
      const auto* rawStrings = strings->data<StringView>();
      const auto delim = delims->valueAt<StringView>(0);
      rows.applyToSelected([&](vector_size_t row) {
        applyInner(rawStrings[row], delim, limit, row, resultWriter);
      });
    } else {
      // The rest of the cases are handled through this general path and no
      // direct access.
      rows.applyToSelected([&](vector_size_t row) {
        applyInner(
            strings->valueAt<StringView>(row),
            delims->valueAt<StringView>(row),
            limit,
            row,
            resultWriter);
      });
    }

    resultWriter.finish();
    // Ensure that our result elements vector uses the same string buffer as
    // the input vector of strings.
    result->as<ArrayVector>()
        ->elements()
        ->as<FlatVector<StringView>>()
        ->acquireSharedStringBuffers(strings->base());
  }

  inline void applyInner(
      StringView input,
      const StringView delim,
      int64_t limit,
      vector_size_t row,
      exec::VectorWriter<Array<Varchar>>& resultWriter) const {
    // Add new array (for the new row) to our array vector.
    resultWriter.setOffset(row);
    auto& arrayWriter = resultWriter.current();

    // Trivial case of converting string to array with 1 element.
    if (limit == 1) {
      arrayWriter.add_item().setNoCopy(input);
      resultWriter.commit();
      return;
    }

    // We walk through our input cutting off the pieces using the delimiter and
    // adding them to the elements vector, until we reached the end of the
    // string or the limit.
    int32_t addedElements{0};
    auto* re = cache_.findOrCompile(delim);
    const auto re2String = re2::StringPiece(input.data(), input.size());
    size_t pos = 0;
    const char* start = input.data();
    re2::StringPiece subMatches[1];
    while (re->Match(
        re2String, pos, input.size(), RE2::Anchor::UNANCHORED, subMatches, 1)) {
      const auto fullMatch = subMatches[0];
      auto offset = fullMatch.data() - start;
      const auto size = fullMatch.size();

      if (size == 0) {
        // delimer is empty string
        offset += 1;
      }

      if (offset >= input.size()) {
        break;
      }

      arrayWriter.add_item().setNoCopy(
          StringView(input.data() + pos, offset - pos));
      pos = offset + size;
      ++addedElements;
      // If the next element should be the last, leave the loop.
      if (addedElements + 1 == limit) {
        break;
      }
    }

    // Add the rest of the string and we are done.
    // Note, that the rest of the string can be empty - we still add it.
    arrayWriter.add_item().setNoCopy(
        StringView(input.data() + pos, input.size() - pos));
    resultWriter.commit();
  }

 private:
  mutable functions::detail::ReCache cache_;
};

std::shared_ptr<exec::VectorFunction> createSplit(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  return std::make_shared<Split>();
}

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  // varchar, varchar -> array(varchar)
  signatures.emplace_back(exec::FunctionSignatureBuilder()
                              .returnType("array(varchar)")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .build());
  signatures.emplace_back(exec::FunctionSignatureBuilder()
                              .returnType("array(varchar)")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .argumentType("bigint")
                              .build());
  return signatures;
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_regexp_split,
    signatures(),
    createSplit);
} // namespace facebook::velox::functions::sparksql
