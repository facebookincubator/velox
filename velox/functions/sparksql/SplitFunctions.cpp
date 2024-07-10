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
#include "velox/functions/prestosql/Utf8Utils.h"

namespace facebook::velox::functions::sparksql {
namespace {

// split(string, delimiter[, limit]) -> array(varchar)
//
// Splits string on delimiter and returns an array of size at most limit.
// delimiter is a string representing regular expression.
// limit is an integer which controls the number of times the regex is applied.
// By default, limit is -1.
class Split final : public exec::VectorFunction {
 public:
  Split() {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);
    DecodedVector* strings = decodedArgs.at(0);
    DecodedVector* delims = decodedArgs.at(1);
    DecodedVector* limits = (args.size() == 2) ? nullptr : decodedArgs.at(2);

    BaseVector::ensureWritable(rows, ARRAY(VARCHAR()), context.pool(), result);
    exec::VectorWriter<Array<Varchar>> resultWriter;
    resultWriter.init(*result->as<ArrayVector>());

    auto getLimit = [&](int32_t row) -> int32_t {
      int32_t limit = std::numeric_limits<int32_t>::max();
      if (limits) {
        const auto limitValue = limits->valueAt<int32_t>(row);
        if (limitValue > 0) {
          limit = limitValue;
        }
      }
      return limit;
    };

    // Fast path for (flat, const, const).
    if (strings->isIdentityMapping() and delims->isConstantMapping() and
        (!limits or limits->isConstantMapping())) {
      const auto* rawStrings = strings->data<StringView>();
      const auto delim = delims->valueAt<StringView>(0);
      auto limit = getLimit(0);
      if (delim.empty()) {
        rows.applyToSelected([&](vector_size_t row) {
          splitEmptyDelimiter(rawStrings[row], limit, row, resultWriter);
        });
      } else {
        rows.applyToSelected([&](vector_size_t row) {
          split(rawStrings[row], delim, limit, row, resultWriter);
        });
      }
    } else {
      rows.applyToSelected([&](vector_size_t row) {
        const auto delim = delims->valueAt<StringView>(row);
        auto limit = getLimit(row);
        if (delim.empty()) {
          splitEmptyDelimiter(
              strings->valueAt<StringView>(row), limit, row, resultWriter);
        } else {
          split(
              strings->valueAt<StringView>(row),
              delim,
              limit,
              row,
              resultWriter);
        }
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

 private:
  // When pattern is empty, split each character out. Since Spark 3.4, when
  // delimiter is empty, the result does not include an empty tail string, e.g.
  // split('abc', '') outputs ["a", "b", "c"] instead of ["a", "b", "c", ""].
  // The result does not include remaining string when limit is smaller than the
  // string size, e.g. split('abc', '', 2) outputs ["a", "b"] instead of ["a",
  // "bc"].
  void splitEmptyDelimiter(
      const StringView current,
      int32_t limit,
      vector_size_t row,
      exec::VectorWriter<Array<Varchar>>& resultWriter) const {
    resultWriter.setOffset(row);
    auto& arrayWriter = resultWriter.current();
    if (current.size() == 0) {
      arrayWriter.add_item().setNoCopy(StringView());
      resultWriter.commit();
      return;
    }

    const size_t end = current.size();
    const char* start = current.data();
    size_t pos = 0;
    int32_t count = 0;
    while (pos < end && count < limit) {
      auto charLength = tryGetCharLength(start + pos, end - pos);
      VELOX_DCHECK_GT(charLength, 0);
      arrayWriter.add_item().setNoCopy(StringView(start + pos, charLength));
      pos += charLength;
      count += 1;
    }
    resultWriter.commit();
  }

  // Split with a non-empty delimiter. If limit > 0, The resulting array's
  // length will not be more than limit and the resulting array's last entry
  // will contain all input beyond the last matched regex. If limit <= 0,
  // delimiter will be applied as many times as possible, and the resulting
  // array can be of any size.
  void split(
      const StringView input,
      const StringView delim,
      int32_t limit,
      vector_size_t row,
      exec::VectorWriter<Array<Varchar>>& resultWriter) const {
    VELOX_USER_CHECK(!delim.empty(), "Non-empty delimiter is expected");
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
    // add them to the elements vector, until we reach the end of the
    // string or the limit.
    int32_t addedElements{0};
    auto* re = cache_.findOrCompile(delim);
    const size_t end = input.size();
    const char* start = input.data();
    const auto re2String = re2::StringPiece(start, end);
    size_t pos = 0;

    re2::StringPiece subMatches[1];
    // Matches a regular expression against a portion of the input string,
    // starting from 'pos' to the end of the input string. The match is not
    // anchored, which means it can start at any position in the string. If a
    // match is found, the matched portion of the string is stored in
    // 'subMatches'. The '1' indicates that we are only interested in the first
    // match found from the current position 'pos' in each iteration of the
    // loop.
    while (re->Match(
        re2String, pos, end, RE2::Anchor::UNANCHORED, subMatches, 1)) {
      const auto fullMatch = subMatches[0];
      auto offset = fullMatch.data() - start;
      const auto size = fullMatch.size();
      if (offset >= end) {
        break;
      }

      // Hit an empty match for the specific case when the delimiter has '|' as
      // a suffix. For example, if the string is 'abc' and the delimiter is 'd|'
      // where '|' is the suffix of the pattern, using the delimiter 'd|' to
      // split "abc" would result in an empty size match in each iteration. For
      // empty matches, instead of inserting an empty character into the result
      // array, always split the character at the current 'pos' of the string
      // and put it into the result array, and then an empty tail string. Thus,
      // the result array for the above example would be ["a", "b", "c", ""].
      if (size == 0) {
        offset += tryGetCharLength(start + pos, end - pos);
      }
      arrayWriter.add_item().setNoCopy(StringView(start + pos, offset - pos));
      pos = offset + size;

      ++addedElements;
      // If the next element should be the last, leave the loop.
      if (addedElements + 1 == limit) {
        break;
      }
    }

    // Add the rest of the string and we are done.
    // Note that the rest of the string can be empty - we still add it.
    arrayWriter.add_item().setNoCopy(StringView(start + pos, end - pos));
    resultWriter.commit();
  }

  mutable functions::detail::ReCache cache_;
};

std::shared_ptr<exec::VectorFunction> createSplit(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK(
      inputArgs.size() == 2 || inputArgs.size() == 3,
      "Two or three arguments are required for split function.");
  VELOX_USER_CHECK(
      inputArgs[0].type->isVarchar(),
      "The first argument should be of varchar type.");
  VELOX_USER_CHECK(
      inputArgs[1].type->isVarchar(),
      "The second argument should be of varchar type.");
  if (inputArgs.size() > 2) {
    VELOX_USER_CHECK(
        inputArgs[2].type->kind() == TypeKind::INTEGER,
        "The third argument should be of integer type.");
  }
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
                              .argumentType("integer")
                              .build());
  return signatures;
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_regexp_split,
    signatures(),
    createSplit);
} // namespace facebook::velox::functions::sparksql
