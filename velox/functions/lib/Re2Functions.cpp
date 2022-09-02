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
#include "velox/functions/lib/Re2Functions.h"

#include <re2/re2.h>
#include <optional>
#include <string>

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/lib/ArrayBuilder.h"
#include "velox/type/StringView.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions {
namespace {

using ::facebook::velox::exec::EvalCtx;
using ::facebook::velox::exec::Expr;
using ::facebook::velox::exec::VectorFunction;
using ::facebook::velox::exec::VectorFunctionArg;
using ::re2::RE2;

std::string printTypesCsv(
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  std::string result;
  result.reserve(inputArgs.size() * 10);
  for (const auto& input : inputArgs) {
    folly::toAppend(
        result.empty() ? "" : ", ", input.type->toString(), &result);
  }
  return result;
}

template <typename T>
re2::StringPiece toStringPiece(const T& s) {
  return re2::StringPiece(s.data(), s.size());
}

// If v is a non-null constant vector, returns the constant value. Otherwise
// returns nullopt.
template <typename T>
std::optional<T> getIfConstant(const BaseVector& v) {
  if (v.encoding() == VectorEncoding::Simple::CONSTANT &&
      v.isNullAt(0) == false) {
    return v.as<ConstantVector<T>>()->valueAt(0);
  }
  return std::nullopt;
}

void checkForBadPattern(const RE2& re) {
  if (UNLIKELY(!re.ok())) {
    VELOX_USER_FAIL("invalid regular expression:{}", re.error());
  }
}

FlatVector<bool>& ensureWritableBool(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  context.ensureWritable(rows, BOOLEAN(), result);
  return *result->as<FlatVector<bool>>();
}

FlatVector<StringView>& ensureWritableStringView(
    const SelectivityVector& rows,
    EvalCtx& context,
    std::shared_ptr<BaseVector>& result) {
  context.ensureWritable(rows, VARCHAR(), result);
  auto* flat = result->as<FlatVector<StringView>>();
  flat->mutableValues(rows.end());
  return *flat;
}

bool re2FullMatch(StringView str, const RE2& re) {
  return RE2::FullMatch(toStringPiece(str), re);
}

bool re2PartialMatch(StringView str, const RE2& re) {
  return RE2::PartialMatch(toStringPiece(str), re);
}

bool re2Extract(
    FlatVector<StringView>& result,
    int row,
    const RE2& re,
    const exec::LocalDecodedVector& strs,
    std::vector<re2::StringPiece>& groups,
    int32_t groupId,
    bool emptyNoMatch) {
  const StringView str = strs->valueAt<StringView>(row);
  DCHECK_GT(groups.size(), groupId);
  if (!re.Match(
          toStringPiece(str),
          0,
          str.size(),
          RE2::UNANCHORED, // Full match not required.
          groups.data(),
          groupId + 1)) {
    if (emptyNoMatch) {
      result.setNoCopy(row, StringView(nullptr, 0));
      return true;
    } else {
      result.setNull(row, true);
      return false;
    }
  } else {
    const re2::StringPiece extracted = groups[groupId];
    result.setNoCopy(row, StringView(extracted.data(), extracted.size()));
    return !StringView::isInline(extracted.size());
  }
}

std::string likePatternToRe2(
    StringView pattern,
    std::optional<char> escapeChar,
    bool& validPattern) {
  std::string regex;
  validPattern = true;
  regex.reserve(pattern.size() * 2);
  regex.append("^");
  bool escaped = false;
  for (const char c : pattern) {
    if (escaped && !(c == '%' || c == '_' || c == escapeChar)) {
      validPattern = false;
    }
    if (!escaped && c == escapeChar) {
      escaped = true;
    } else {
      switch (c) {
        case '%':
          regex.append(escaped ? "%" : ".*");
          escaped = false;
          break;
        case '_':
          regex.append(escaped ? "_" : ".");
          escaped = false;
          break;
        // Escape all the meta characters in re2
        case '\\':
        case '|':
        case '^':
        case '$':
        case '.':
        case '*':
        case '+':
        case '?':
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
          regex.append("\\");
        // Append the meta character after the escape. Note: The fallthrough is
        // intentional.
        default:
          regex.append(1, c);
          escaped = false;
      }
    }
  }
  if (escaped) {
    validPattern = false;
  }

  regex.append("$");
  return regex;
}

template <bool (*Fn)(StringView, const RE2&)>
class Re2MatchConstantPattern final : public VectorFunction {
 public:
  explicit Re2MatchConstantPattern(StringView pattern)
      : re_(toStringPiece(pattern), RE2::Quiet) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK_EQ(args.size(), 2);
    FlatVector<bool>& result = ensureWritableBool(rows, context, resultRef);
    exec::LocalDecodedVector toSearch(context, *args[0], rows);
    checkForBadPattern(re_);
    rows.applyToSelected([&](vector_size_t i) {
      result.set(i, Fn(toSearch->valueAt<StringView>(i), re_));
    });
  }

 private:
  RE2 re_;
};

template <bool (*Fn)(StringView, const RE2&)>
class Re2Match final : public VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx& context,
      VectorPtr& resultRef) const override {
    VELOX_CHECK_EQ(args.size(), 2);
    if (auto pattern = getIfConstant<StringView>(*args[1])) {
      Re2MatchConstantPattern<Fn>(*pattern).apply(
          rows, args, outputType, context, resultRef);
      return;
    }
    // General case.
    FlatVector<bool>& result = ensureWritableBool(rows, context, resultRef);
    exec::LocalDecodedVector toSearch(context, *args[0], rows);
    exec::LocalDecodedVector pattern(context, *args[1], rows);
    rows.applyToSelected([&](vector_size_t row) {
      RE2 re(toStringPiece(pattern->valueAt<StringView>(row)), RE2::Quiet);
      checkForBadPattern(re);
      result.set(row, Fn(toSearch->valueAt<StringView>(row), re));
    });
  }
};

void checkForBadGroupId(int groupId, const RE2& re) {
  if (UNLIKELY(groupId < 0 || groupId > re.NumberOfCapturingGroups())) {
    VELOX_USER_FAIL("No group {} in regex '{}'", groupId, re.pattern());
  }
}

template <typename T>
class Re2SearchAndExtractConstantPattern final : public VectorFunction {
 public:
  explicit Re2SearchAndExtractConstantPattern(
      StringView pattern,
      bool emptyNoMatch)
      : re_(toStringPiece(pattern), RE2::Quiet), emptyNoMatch_(emptyNoMatch) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK(args.size() == 2 || args.size() == 3);
    // TODO: Potentially re-use the string vector, not just the buffer.
    FlatVector<StringView>& result =
        ensureWritableStringView(rows, context, resultRef);

    // apply() will not be invoked if the selection is empty.
    checkForBadPattern(re_);

    exec::LocalDecodedVector toSearch(context, *args[0], rows);
    bool mustRefSourceStrings = false;
    FOLLY_DECLARE_REUSED(groups, std::vector<re2::StringPiece>);
    // Common case: constant group id.
    if (args.size() == 2) {
      groups.resize(1);
      rows.applyToSelected([&](vector_size_t i) {
        mustRefSourceStrings |=
            re2Extract(result, i, re_, toSearch, groups, 0, emptyNoMatch_);
      });
      if (mustRefSourceStrings) {
        result.acquireSharedStringBuffers(toSearch->base());
      }
      return;
    }

    if (const auto groupId = getIfConstant<T>(*args[2])) {
      checkForBadGroupId(*groupId, re_);
      groups.resize(*groupId + 1);
      rows.applyToSelected([&](vector_size_t i) {
        mustRefSourceStrings |= re2Extract(
            result, i, re_, toSearch, groups, *groupId, emptyNoMatch_);
      });
      if (mustRefSourceStrings) {
        result.acquireSharedStringBuffers(toSearch->base());
      }
      return;
    }

    // Less common case: variable group id. Resize the groups vector to
    // accommodate the largest group id referenced.
    exec::LocalDecodedVector groupIds(context, *args[2], rows);
    T maxGroupId = 0, minGroupId = 0;
    rows.applyToSelected([&](vector_size_t i) {
      maxGroupId = std::max(groupIds->valueAt<T>(i), maxGroupId);
      minGroupId = std::min(groupIds->valueAt<T>(i), minGroupId);
    });
    checkForBadGroupId(maxGroupId, re_);
    checkForBadGroupId(minGroupId, re_);
    groups.resize(maxGroupId + 1);
    rows.applyToSelected([&](vector_size_t i) {
      T group = groupIds->valueAt<T>(i);
      mustRefSourceStrings |=
          re2Extract(result, i, re_, toSearch, groups, group, emptyNoMatch_);
    });
    if (mustRefSourceStrings) {
      result.acquireSharedStringBuffers(toSearch->base());
    }
  }

 private:
  RE2 re_;
  const bool emptyNoMatch_;
}; // namespace

// The factory function we provide returns a unique instance for each call, so
// this is safe.
template <typename T>
class Re2SearchAndExtract final : public VectorFunction {
 public:
  explicit Re2SearchAndExtract(bool emptyNoMatch)
      : emptyNoMatch_(emptyNoMatch) {}
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK(args.size() == 2 || args.size() == 3);
    // Handle the common case of a constant pattern.
    if (auto pattern = getIfConstant<StringView>(*args[1])) {
      Re2SearchAndExtractConstantPattern<T>(*pattern, emptyNoMatch_)
          .apply(rows, args, outputType, context, resultRef);
      return;
    }

    // The general case. Further optimizations are possible to avoid regex
    // recompilation, but a constant pattern is by far the most common case.
    FlatVector<StringView>& result =
        ensureWritableStringView(rows, context, resultRef);
    exec::LocalDecodedVector toSearch(context, *args[0], rows);
    exec::LocalDecodedVector pattern(context, *args[1], rows);
    bool mustRefSourceStrings = false;
    FOLLY_DECLARE_REUSED(groups, std::vector<re2::StringPiece>);
    if (args.size() == 2) {
      groups.resize(1);
      rows.applyToSelected([&](vector_size_t i) {
        RE2 re(toStringPiece(pattern->valueAt<StringView>(i)), RE2::Quiet);
        checkForBadPattern(re);
        mustRefSourceStrings |=
            re2Extract(result, i, re, toSearch, groups, 0, emptyNoMatch_);
      });
    } else {
      exec::LocalDecodedVector groupIds(context, *args[2], rows);
      rows.applyToSelected([&](vector_size_t i) {
        const auto groupId = groupIds->valueAt<T>(i);
        RE2 re(toStringPiece(pattern->valueAt<StringView>(i)), RE2::Quiet);
        checkForBadPattern(re);
        checkForBadGroupId(groupId, re);
        groups.resize(groupId + 1);
        mustRefSourceStrings |=
            re2Extract(result, i, re, toSearch, groups, groupId, emptyNoMatch_);
      });
    }
    if (mustRefSourceStrings) {
      result.acquireSharedStringBuffers(toSearch->base());
    }
  }

 private:
  const bool emptyNoMatch_;
};

// Match string 'input' with a fixed pattern (with no wildcard characters).
bool matchExactPattern(
    StringView input,
    StringView pattern,
    vector_size_t length) {
  return input.size() == pattern.size() &&
      std::memcmp(input.data(), pattern.data(), length) == 0;
}

// Match the first 'length' characters of string 'input' and prefix pattern.
bool matchPrefixPattern(
    StringView input,
    StringView pattern,
    vector_size_t length) {
  return input.size() >= length &&
      std::memcmp(input.data(), pattern.data(), length) == 0;
}

// Match the last 'length' characters of string 'input' and suffix pattern.
bool matchSuffixPattern(
    StringView input,
    StringView pattern,
    vector_size_t length) {
  return input.size() >= length &&
      std::memcmp(
          input.data() + input.size() - length,
          pattern.data() + pattern.size() - length,
          length) == 0;
}

// Search for 'pattern' in 'input' starting from positions greater than
// 'startPosition'.
vector_size_t matchPatternAtPosition(
    const StringView& input,
    const StringView& pattern,
    vector_size_t startPosition) {
  const auto inputData = input.data();
  const auto patternData = pattern.data();
  const auto patternSize = pattern.size();
  if (input.size() - startPosition < patternSize) {
    return -1;
  }
  const auto lastPosition = input.size() - patternSize;

  auto i = startPosition;
  while (i <= lastPosition) {
    if (std::memcmp(inputData + i, patternData, patternSize) == 0) {
      return i;
    }
    i++;
  }
  return -1;
}

template <PatternKind P>
class OptimizedLikeWithMemcmp final : public VectorFunction {
 public:
  OptimizedLikeWithMemcmp(
      const StringView& pattern,
      const PatternMetadata& patternParameters)
      : pattern_{pattern}, patternMetadata_{patternParameters} {
    trailingWildcard_ = pattern_.data()[pattern_.size() - 1] == '%';
  }

  // Match 'numFixedPatterns' fixed patterns from the vector of fixedPatterns
  // in the pattern with input in their order of occurence.
  vector_size_t matchFixedPatterns(
      const StringView& input,
      const vector_size_t numMatches) const {
    const auto fixedPatterns = patternMetadata_.fixedPatterns.value();
    auto startPosition = 0;

    for (auto i = 0; i < numMatches; i++) {
      const std::string fixedPatternString{fixedPatterns[i]};
      auto fixedPattern = StringView(fixedPatternString);
      if ((startPosition = matchPatternAtPosition(
               input, fixedPattern, startPosition)) == -1) {
        return -1;
      }
      startPosition += fixedPattern.size();
    }
    return startPosition;
  }

  // Match all fixed patterns in an input string having fixed patterns
  // interspersed with one or more '%' wildcard character streams.
  bool matchFixedWithWildcard(const StringView& input) const {
    auto startPosition = 0;
    const auto numFixedPatterns = trailingWildcard_
        ? patternMetadata_.numFixedPatterns
        : patternMetadata_.numFixedPatterns - 1;
    if ((startPosition = matchFixedPatterns(input, numFixedPatterns)) == -1) {
      return false;
    }

    // Match the last fixed pattern with memcmp when there is no trailing '%'
    // wildcard character.
    if (!trailingWildcard_) {
      const std::string lastPatternString{
          patternMetadata_.fixedPatterns
              .value()[patternMetadata_.numFixedPatterns - 1]};
      auto lastPattern = StringView(lastPatternString);
      auto fixedPatternLength = lastPattern.size();
      return (
          input.size() - fixedPatternLength >= startPosition &&
          std::memcmp(
              input.data() + input.size() - fixedPatternLength,
              lastPattern.data(),
              fixedPatternLength) == 0);
    }
    return true;
  }

  bool match(StringView input) const {
    switch (P) {
      case PatternKind::kExactlyN:
        return input.size() == patternMetadata_.numSingleWildcards;
      case PatternKind::kAtLeastN:
        return input.size() >= patternMetadata_.numSingleWildcards;
      case PatternKind::kFixed:
        return matchExactPattern(
            input, pattern_, patternMetadata_.reducedPatternLength);
      case PatternKind::kPrefix:
        return matchPrefixPattern(
            input, pattern_, patternMetadata_.reducedPatternLength);
      case PatternKind::kSuffix:
        return matchSuffixPattern(
            input, pattern_, patternMetadata_.reducedPatternLength);
      case PatternKind::kMiddleWildcards:
        return matchFixedWithWildcard(input);
    }
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK(args.size() == 2 || args.size() == 3);
    FlatVector<bool>& result = ensureWritableBool(rows, context, resultRef);
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto toSearch = decodedArgs.at(0);

    if (toSearch->isIdentityMapping()) {
      auto input = toSearch->data<StringView>();
      rows.applyToSelected(
          [&](vector_size_t i) { result.set(i, match(input[i])); });
      return;
    }
    if (toSearch->isConstantMapping()) {
      auto input = toSearch->valueAt<StringView>(0);
      bool matchResult = match(input);
      rows.applyToSelected(
          [&](vector_size_t i) { result.set(i, matchResult); });
      return;
    }

    // Since the likePattern and escapeChar (2nd and 3rd args) are both
    // constants, so the first arg is expected to be either of flat or constant
    // vector only. This code path is unreachable.
    VELOX_UNREACHABLE();
  }

 private:
  const StringView pattern_;
  const PatternMetadata patternMetadata_;
  bool trailingWildcard_;
};

class LikeWithRe2 final : public VectorFunction {
 public:
  LikeWithRe2(StringView pattern, std::optional<char> escapeChar) {
    RE2::Options opt{RE2::Quiet};
    opt.set_dot_nl(true);
    re_.emplace(
        toStringPiece(likePatternToRe2(pattern, escapeChar, validPattern_)),
        opt);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK(args.size() == 2 || args.size() == 3);

    if (!validPattern_) {
      auto error = std::make_exception_ptr(std::invalid_argument(
          "Escape character must be followed by '%%', '_' or the escape character itself\""));
      rows.applyToSelected([&](auto row) { context.setError(row, error); });
      return;
    }

    // apply() will not be invoked if the selection is empty.
    checkForBadPattern(*re_);
    FlatVector<bool>& result = ensureWritableBool(rows, context, resultRef);

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto toSearch = decodedArgs.at(0);
    if (toSearch->isIdentityMapping()) {
      auto rawStrings = toSearch->data<StringView>();
      rows.applyToSelected([&](vector_size_t i) {
        result.set(i, re2FullMatch(rawStrings[i], *re_));
      });
      return;
    }

    if (toSearch->isConstantMapping()) {
      bool match = re2FullMatch(toSearch->valueAt<StringView>(0), *re_);
      rows.applyToSelected([&](vector_size_t i) { result.set(i, match); });
      return;
    }

    // Since the likePattern and escapeChar (2nd and 3rd args) are both
    // constants, so the first arg is expected to be either of flat or constant
    // vector only. This code path is unreachable.
    VELOX_UNREACHABLE();
  }

 private:
  std::optional<RE2> re_;
  bool validPattern_;
};

void re2ExtractAll(
    ArrayBuilder<Varchar>& builder,
    const RE2& re,
    const exec::LocalDecodedVector& inputStrs,
    const int row,
    std::vector<re2::StringPiece>& groups,
    int32_t groupId) {
  ArrayBuilder<Varchar>::Ref array = builder.startArray(row);
  const StringView str = inputStrs->valueAt<StringView>(row);
  const re2::StringPiece input = toStringPiece(str);
  size_t pos = 0;

  while (re.Match(
      input, pos, input.size(), RE2::UNANCHORED, groups.data(), groupId + 1)) {
    DCHECK_GT(groups.size(), groupId);

    const re2::StringPiece fullMatch = groups[0];
    const re2::StringPiece subMatch = groups[groupId];

    array.emplace_back(subMatch.data(), subMatch.size());
    pos = fullMatch.data() + fullMatch.size() - input.data();
    if (UNLIKELY(fullMatch.size() == 0)) {
      ++pos;
    }
  }
}

template <typename T>
class Re2ExtractAllConstantPattern final : public VectorFunction {
 public:
  explicit Re2ExtractAllConstantPattern(StringView pattern)
      : re_(toStringPiece(pattern), RE2::Quiet) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK(args.size() == 2 || args.size() == 3);
    checkForBadPattern(re_);

    ArrayBuilder<Varchar> builder(
        rows.size(), rows.countSelected() * 3, context.pool());
    exec::LocalDecodedVector inputStrs(context, *args[0], rows);
    FOLLY_DECLARE_REUSED(groups, std::vector<re2::StringPiece>);

    if (args.size() == 2) {
      // Case 1: No groupId -- use 0 as the default groupId
      //
      groups.resize(1);
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        re2ExtractAll(builder, re_, inputStrs, row, groups, 0);
      });
    } else if (const auto _groupId = getIfConstant<T>(*args[2])) {
      // Case 2: Constant groupId
      //
      checkForBadGroupId(*_groupId, re_);
      groups.resize(*_groupId + 1);
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        re2ExtractAll(builder, re_, inputStrs, row, groups, *_groupId);
      });
    } else {
      // Case 3: Variable groupId, so resize the groups vector to accommodate
      // the largest group id referenced.
      //
      exec::LocalDecodedVector groupIds(context, *args[2], rows);
      T maxGroupId = 0, minGroupId = 0;
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        maxGroupId = std::max(groupIds->valueAt<T>(row), maxGroupId);
        minGroupId = std::min(groupIds->valueAt<T>(row), minGroupId);
      });
      checkForBadGroupId(maxGroupId, re_);
      checkForBadGroupId(minGroupId, re_);
      groups.resize(maxGroupId + 1);
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        const T groupId = groupIds->valueAt<T>(row);
        checkForBadGroupId(groupId, re_);
        re2ExtractAll(builder, re_, inputStrs, row, groups, groupId);
      });
    }

    if (const auto fv = inputStrs->base()->asFlatVector<StringView>()) {
      builder.setStringBuffers(fv->stringBuffers());
    }
    std::shared_ptr<ArrayVector> arrayVector =
        std::move(builder).finish(context.pool());
    context.moveOrCopyResult(arrayVector, rows, resultRef);
  }

 private:
  RE2 re_;
};

template <typename T>
class Re2ExtractAll final : public VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK(args.size() == 2 || args.size() == 3);
    // Use Re2ExtractAllConstantPattern if it's constant regexp pattern.
    //
    if (auto pattern = getIfConstant<StringView>(*args[1])) {
      Re2ExtractAllConstantPattern<T>(*pattern).apply(
          rows, args, outputType, context, resultRef);
      return;
    }

    ArrayBuilder<Varchar> builder(
        rows.size(), rows.countSelected() * 3, context.pool());
    exec::LocalDecodedVector inputStrs(context, *args[0], rows);
    exec::LocalDecodedVector pattern(context, *args[1], rows);
    FOLLY_DECLARE_REUSED(groups, std::vector<re2::StringPiece>);

    if (args.size() == 2) {
      // Case 1: No groupId -- use 0 as the default groupId
      //
      groups.resize(1);
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        RE2 re(toStringPiece(pattern->valueAt<StringView>(row)), RE2::Quiet);
        checkForBadPattern(re);
        re2ExtractAll(builder, re, inputStrs, row, groups, 0);
      });
    } else {
      // Case 2: Has groupId
      //
      exec::LocalDecodedVector groupIds(context, *args[2], rows);
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        const T groupId = groupIds->valueAt<T>(row);
        RE2 re(toStringPiece(pattern->valueAt<StringView>(row)), RE2::Quiet);
        checkForBadPattern(re);
        checkForBadGroupId(groupId, re);
        groups.resize(groupId + 1);
        re2ExtractAll(builder, re, inputStrs, row, groups, groupId);
      });
    }

    if (const auto fv = inputStrs->base()->asFlatVector<StringView>()) {
      builder.setStringBuffers(fv->stringBuffers());
    }
    std::shared_ptr<ArrayVector> arrayVector =
        std::move(builder).finish(context.pool());
    context.moveOrCopyResult(arrayVector, rows, resultRef);
  }
};

template <bool (*Fn)(StringView, const RE2&)>
std::shared_ptr<VectorFunction> makeRe2MatchImpl(
    const std::string& name,
    const std::vector<VectorFunctionArg>& inputArgs) {
  if (inputArgs.size() != 2 || !inputArgs[0].type->isVarchar() ||
      !inputArgs[1].type->isVarchar()) {
    VELOX_UNSUPPORTED(
        "{} expected (VARCHAR, VARCHAR) but got ({})",
        name,
        printTypesCsv(inputArgs));
  }

  BaseVector* constantPattern = inputArgs[1].constantValue.get();

  if (constantPattern != nullptr && !constantPattern->isNullAt(0)) {
    return std::make_shared<Re2MatchConstantPattern<Fn>>(
        constantPattern->as<ConstantVector<StringView>>()->valueAt(0));
  }
  static std::shared_ptr<Re2Match<Fn>> kMatchExpr =
      std::make_shared<Re2Match<Fn>>();
  return kMatchExpr;
}

} // namespace

std::shared_ptr<VectorFunction> makeRe2Match(
    const std::string& name,
    const std::vector<VectorFunctionArg>& inputArgs) {
  return makeRe2MatchImpl<re2FullMatch>(name, inputArgs);
}

std::vector<std::shared_ptr<exec::FunctionSignature>> re2MatchSignatures() {
  // varchar, varchar -> boolean
  return {exec::FunctionSignatureBuilder()
              .returnType("boolean")
              .argumentType("varchar")
              .argumentType("varchar")
              .build()};
}

std::shared_ptr<VectorFunction> makeRe2Search(
    const std::string& name,
    const std::vector<VectorFunctionArg>& inputArgs) {
  return makeRe2MatchImpl<re2PartialMatch>(name, inputArgs);
}

std::vector<std::shared_ptr<exec::FunctionSignature>> re2SearchSignatures() {
  // varchar, varchar -> boolean
  return {exec::FunctionSignatureBuilder()
              .returnType("boolean")
              .argumentType("varchar")
              .argumentType("varchar")
              .build()};
}

std::shared_ptr<VectorFunction> makeRe2Extract(
    const std::string& name,
    const std::vector<VectorFunctionArg>& inputArgs,
    const bool emptyNoMatch) {
  auto numArgs = inputArgs.size();
  VELOX_USER_CHECK(
      numArgs == 2 || numArgs == 3,
      "{} requires 2 or 3 arguments, but got {}",
      name,
      numArgs);

  VELOX_USER_CHECK(
      inputArgs[0].type->isVarchar(),
      "{} requires first argument of type VARCHAR, but got {}",
      name,
      inputArgs[0].type->toString());

  VELOX_USER_CHECK(
      inputArgs[1].type->isVarchar(),
      "{} requires second argument of type VARCHAR, but got {}",
      name,
      inputArgs[1].type->toString());

  TypeKind groupIdTypeKind = TypeKind::INTEGER;
  if (numArgs == 3) {
    groupIdTypeKind = inputArgs[2].type->kind();
    VELOX_USER_CHECK(
        groupIdTypeKind == TypeKind::INTEGER ||
            groupIdTypeKind == TypeKind::BIGINT,
        "{} requires third argument of type INTEGER or BIGINT, but got {}",
        name,
        mapTypeKindToName(groupIdTypeKind));
  }

  BaseVector* constantPattern = inputArgs[1].constantValue.get();

  if (constantPattern != nullptr && !constantPattern->isNullAt(0)) {
    auto pattern =
        constantPattern->as<ConstantVector<StringView>>()->valueAt(0);
    switch (groupIdTypeKind) {
      case TypeKind::INTEGER:
        return std::make_shared<Re2SearchAndExtractConstantPattern<int32_t>>(
            pattern, emptyNoMatch);
      case TypeKind::BIGINT:
        return std::make_shared<Re2SearchAndExtractConstantPattern<int64_t>>(
            pattern, emptyNoMatch);
      default:
        VELOX_UNREACHABLE();
    }
  }

  switch (groupIdTypeKind) {
    case TypeKind::INTEGER:
      return std::make_shared<Re2SearchAndExtract<int32_t>>(emptyNoMatch);
    case TypeKind::BIGINT:
      return std::make_shared<Re2SearchAndExtract<int64_t>>(emptyNoMatch);
    default:
      VELOX_UNREACHABLE();
  }
}

std::vector<std::shared_ptr<exec::FunctionSignature>> re2ExtractSignatures() {
  // varchar, varchar -> boolean
  // varchar, varchar, integer|bigint -> boolean
  return {
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .argumentType("varchar")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .argumentType("varchar")
          .argumentType("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .argumentType("varchar")
          .argumentType("integer")
          .build(),
  };
}

PatternMetadata determinePatternKind(const StringView& pattern) {
  auto i = 0;
  const auto patternLength = pattern.size();
  const auto patternString = pattern.str();
  auto wildcardStart = 0;
  auto numAnyWildcardStream = 0;
  auto numWildcards = 0;
  auto numAnyWildcards = 0;
  auto numSingleWildcards = 0;
  auto fixedPatternStart = 0;
  auto numFixedPatterns = 0;
  std::vector<std::string> fixedPatterns;

  while (i < patternLength) {
    if (patternString[i] == '%' || patternString[i] == '_') {
      // Look till the last contiguous wildcard character, starting from this
      // index, is found, or the end of pattern is reached.
      wildcardStart = i;
      while (i < patternLength &&
             (patternString[i] == '%' || patternString[i] == '_')) {
        numSingleWildcards += (patternString[i] == '_');
        numAnyWildcards += (patternString[i] == '%');
        i++;
      }
      numAnyWildcardStream += (numAnyWildcards > 0);
    } else {
      // Look till the end of fixed pattern, starting from this index, is found,
      // or the end of pattern is reached.
      fixedPatternStart = i;
      while (i < patternLength &&
             (patternString[i] != '%' && patternString[i] != '_')) {
        i++;
      }
      numFixedPatterns++;
      fixedPatterns.emplace_back(
          patternString.substr(fixedPatternStart, i - fixedPatternStart));
    }
  }
  numWildcards = numSingleWildcards + numAnyWildcards;

  // Pattern contains only wildcard characters.
  if (!numFixedPatterns && numWildcards) {
    if (!numAnyWildcards) {
      return PatternMetadata(PatternKind::kExactlyN, numSingleWildcards, 0, 0);
    }
    return PatternMetadata(PatternKind::kAtLeastN, numSingleWildcards, 0, 0);
  }
  // Pattern is fixed if there are no wildcards.
  if (!numWildcards && numFixedPatterns == 1) {
    return PatternMetadata(PatternKind::kFixed, 0, patternLength, 0);
  }
  // Patterns containing one fixed pattern and a single stream of '%' wildcards
  // is either a prefix or a suffix pattern. Based on the positions of the fixed
  // pattern and contiguous wildcard character stream, classify accordingly.
  if (!numSingleWildcards && numFixedPatterns <= 1 &&
      numAnyWildcardStream <= 1) {
    if (fixedPatternStart < wildcardStart) {
      return PatternMetadata(PatternKind::kPrefix, 0, wildcardStart, 0);
    }
    return PatternMetadata(
        PatternKind::kSuffix, 0, patternLength - fixedPatternStart, 0);
  }
  // Patterns with multiple fixed patterns and '%' wildcard character streams
  // are classified as kMiddleWildcard patterns.
  if (!numSingleWildcards && numAnyWildcardStream && numFixedPatterns) {
    return PatternMetadata(
        PatternKind::kMiddleWildcards, 0, 0, numFixedPatterns, fixedPatterns);
  }
  return PatternMetadata(PatternKind::kGeneric, 0, 0, 0);
}

std::shared_ptr<exec::VectorFunction> makeLike(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  auto numArgs = inputArgs.size();
  VELOX_USER_CHECK(
      numArgs == 2 || numArgs == 3,
      "{} requires 2 or 3 arguments, but got {}",
      name,
      numArgs);

  VELOX_USER_CHECK(
      inputArgs[0].type->isVarchar(),
      "{} requires first argument of type VARCHAR, but got {}",
      name,
      inputArgs[0].type->toString());

  VELOX_USER_CHECK(
      inputArgs[1].type->isVarchar(),
      "{} requires second argument of type VARCHAR, but got {}",
      name,
      inputArgs[1].type->toString());

  std::optional<char> escapeChar;
  if (numArgs == 3) {
    VELOX_USER_CHECK(
        inputArgs[2].type->isVarchar(),
        "{} requires third argument of type VARCHAR, but got {}",
        name,
        inputArgs[2].type->toString());

    BaseVector* escape = inputArgs[2].constantValue.get();
    VELOX_USER_CHECK(
        escape != nullptr,
        "{} requires third argument to be a constant of type VARCHAR",
        name,
        inputArgs[2].type->toString());

    auto constantEscape = escape->as<ConstantVector<StringView>>();

    // Escape char should be a single char value
    VELOX_USER_CHECK_EQ(constantEscape->valueAt(0).size(), 1);
    escapeChar = constantEscape->valueAt(0).data()[0];
  }

  BaseVector* constantPattern = inputArgs[1].constantValue.get();
  VELOX_USER_CHECK(
      constantPattern != nullptr,
      "{} requires second argument to be a constant of type VARCHAR",
      name,
      inputArgs[1].type->toString());
  auto pattern = constantPattern->as<ConstantVector<StringView>>()->valueAt(0);
  if (!escapeChar) {
    auto patternMetadata = determinePatternKind(pattern);

    switch (patternMetadata.patternKind) {
      case PatternKind::kExactlyN:
        return std::make_shared<
            OptimizedLikeWithMemcmp<PatternKind::kExactlyN>>(
            pattern, patternMetadata);
      case PatternKind::kAtLeastN:
        return std::make_shared<
            OptimizedLikeWithMemcmp<PatternKind::kAtLeastN>>(
            pattern, patternMetadata);
      case PatternKind::kFixed:
        return std::make_shared<OptimizedLikeWithMemcmp<PatternKind::kFixed>>(
            pattern, patternMetadata);
      case PatternKind::kPrefix:
        return std::make_shared<OptimizedLikeWithMemcmp<PatternKind::kPrefix>>(
            pattern, patternMetadata);
      case PatternKind::kSuffix:
        return std::make_shared<OptimizedLikeWithMemcmp<PatternKind::kSuffix>>(
            pattern, patternMetadata);
      case PatternKind::kMiddleWildcards:
        return std::make_shared<
            OptimizedLikeWithMemcmp<PatternKind::kMiddleWildcards>>(
            pattern, patternMetadata);
      default:
        return std::make_shared<LikeWithRe2>(pattern, escapeChar);
    }
  }
  return std::make_shared<LikeWithRe2>(pattern, escapeChar);
}

std::vector<std::shared_ptr<exec::FunctionSignature>> likeSignatures() {
  // varchar, varchar -> boolean
  // varchar, varchar, varchar -> boolean
  return {
      exec::FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("varchar")
          .argumentType("varchar")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("varchar")
          .argumentType("varchar")
          .argumentType("varchar")
          .build(),
  };
}

std::shared_ptr<VectorFunction> makeRe2ExtractAll(
    const std::string& name,
    const std::vector<VectorFunctionArg>& inputArgs) {
  auto numArgs = inputArgs.size();
  VELOX_USER_CHECK(
      numArgs == 2 || numArgs == 3,
      "{} requires 2 or 3 arguments, but got {}",
      name,
      numArgs);

  VELOX_USER_CHECK(
      inputArgs[0].type->isVarchar(),
      "{} requires first argument of type VARCHAR, but got {}",
      name,
      inputArgs[0].type->toString());

  VELOX_USER_CHECK(
      inputArgs[1].type->isVarchar(),
      "{} requires second argument of type VARCHAR, but got {}",
      name,
      inputArgs[1].type->toString());

  TypeKind groupIdTypeKind = TypeKind::INTEGER;
  if (numArgs == 3) {
    groupIdTypeKind = inputArgs[2].type->kind();
    VELOX_USER_CHECK(
        groupIdTypeKind == TypeKind::INTEGER ||
            groupIdTypeKind == TypeKind::BIGINT,
        "{} requires third argument of type INTEGER or BIGINT, but got {}",
        name,
        mapTypeKindToName(groupIdTypeKind));
  }

  BaseVector* constantPattern = inputArgs[1].constantValue.get();
  if (constantPattern != nullptr && !constantPattern->isNullAt(0)) {
    auto pattern =
        constantPattern->as<ConstantVector<StringView>>()->valueAt(0);
    switch (groupIdTypeKind) {
      case TypeKind::INTEGER:
        return std::make_shared<Re2ExtractAllConstantPattern<int32_t>>(pattern);
      case TypeKind::BIGINT:
        return std::make_shared<Re2ExtractAllConstantPattern<int64_t>>(pattern);
      default:
        VELOX_UNREACHABLE();
    }
  }

  switch (groupIdTypeKind) {
    case TypeKind::INTEGER:
      return std::make_shared<Re2ExtractAll<int32_t>>();
    case TypeKind::BIGINT:
      return std::make_shared<Re2ExtractAll<int64_t>>();
    default:
      VELOX_UNREACHABLE();
  }
}

std::vector<std::shared_ptr<exec::FunctionSignature>>
re2ExtractAllSignatures() {
  // varchar, varchar -> array<varchar>
  // varchar, varchar, integer|bigint -> array<varchar>
  return {
      exec::FunctionSignatureBuilder()
          .returnType("array(varchar)")
          .argumentType("varchar")
          .argumentType("varchar")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(varchar)")
          .argumentType("varchar")
          .argumentType("varchar")
          .argumentType("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(varchar)")
          .argumentType("varchar")
          .argumentType("varchar")
          .argumentType("integer")
          .build(),
  };
}

} // namespace facebook::velox::functions
