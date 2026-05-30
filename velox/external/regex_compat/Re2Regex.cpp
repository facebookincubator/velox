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
#include "velox/external/regex_compat/Re2Regex.h"

#include <re2/re2.h>
#include <re2/stringpiece.h>

#include "velox/functions/lib/Re2Functions.h"
#include "velox/type/StringView.h"

namespace facebook::velox::regex_compat {
namespace {

inline re2::StringPiece toSp(std::string_view s) {
  return re2::StringPiece(s.data(), s.size());
}
inline std::string_view toSv(const re2::StringPiece& sp) {
  return std::string_view(sp.data(), sp.size());
}
inline StringView toVelox(std::string_view s) {
  return StringView(s.data(), s.size());
}

re2::RE2::Anchor toRe2Anchor(Anchor a) {
  switch (a) {
    case Anchor::kUnanchored:
      return re2::RE2::UNANCHORED;
    case Anchor::kAnchorStart:
      return re2::RE2::ANCHOR_START;
    case Anchor::kAnchorBoth:
      return re2::RE2::ANCHOR_BOTH;
  }
  return re2::RE2::UNANCHORED;
}

re2::RE2::Options toRe2Options(const Options& o) {
  re2::RE2::Options out;
  out.set_case_sensitive(o.caseSensitive);
  out.set_dot_nl(o.dotNl);
  out.set_one_line(o.oneLine);
  out.set_log_errors(o.logErrors);
  out.set_max_mem(o.maxMem);
  out.set_encoding(re2::RE2::Options::EncodingUTF8);
  return out;
}

} // namespace

Re2Regex::Re2Regex(std::string_view javaPattern, Options opt) {
  // Translate Java syntax to RE2 syntax using the same helper Velox
  // Spark/Presto functions use.  This is a free function in Re2Functions.h
  // (FOLLY_ALWAYS_INLINE), so there's no separate dependency.
  std::string re2Pattern =
      functions::prepareRegexpReplacePattern(toVelox(javaPattern));
  // Java's MULTILINE flag doesn't map cleanly to any RE2 Options bit:
  // RE2's default behavior is that `^` and `$` only match at the start/end
  // of the entire input.  The inline `(?m)` modifier is the only way to
  // enable per-line anchoring.  We prepend it when the caller asks for
  // MULTILINE (oneLine == false).  Java MULTILINE is purely additive
  // (it doesn't affect `.` or non-anchor metas), so prepending is safe.
  if (!opt.oneLine) {
    re2Pattern = "(?m)" + re2Pattern;
  }
  re_ = std::make_unique<re2::RE2>(toSp(re2Pattern), toRe2Options(opt));
  if (!re_->ok()) {
    error_ = re_->error();
    return;
  }
  named_ = re_->NamedCapturingGroups();
}

Re2Regex::~Re2Regex() = default;

bool Re2Regex::ok() const {
  return re_ && re_->ok();
}
const std::string& Re2Regex::error() const {
  return error_;
}
int Re2Regex::NumberOfCapturingGroups() const {
  return re_ ? re_->NumberOfCapturingGroups() : 0;
}
const std::map<std::string, int>& Re2Regex::NamedCapturingGroups() const {
  return named_;
}
const re2::RE2& Re2Regex::raw() const {
  return *re_;
}

bool Re2Regex::Match(
    std::string_view input,
    std::size_t startpos,
    std::size_t endpos,
    Anchor anchor,
    std::string_view* submatch,
    int nsubmatch) const {
  if (!ok()) {
    return false;
  }
  // RE2 writes into StringPiece buffer; copy into caller's string_view array.
  std::vector<re2::StringPiece> caps(nsubmatch);
  bool matched = re_->Match(
      toSp(input),
      startpos,
      endpos,
      toRe2Anchor(anchor),
      caps.data(),
      nsubmatch);
  if (!matched) {
    return false;
  }
  for (int i = 0; i < nsubmatch; ++i) {
    submatch[i] = caps[i].data() ? toSv(caps[i]) : std::string_view{};
  }
  return true;
}

bool Re2Regex::FullMatch(std::string_view input, const Re2Regex& re) {
  if (!re.ok()) {
    return false;
  }
  return re2::RE2::FullMatch(toSp(input), *re.re_);
}

bool Re2Regex::PartialMatch(std::string_view input, const Re2Regex& re) {
  if (!re.ok()) {
    return false;
  }
  return re2::RE2::PartialMatch(toSp(input), *re.re_);
}

int Re2Regex::GlobalReplace(
    std::string* str,
    const Re2Regex& re,
    std::string_view javaReplacement) {
  if (!re.ok() || str == nullptr) {
    return 0;
  }
  const std::string re2Replacement = functions::prepareRegexpReplaceReplacement(
      *re.re_, toVelox(javaReplacement));
  return re2::RE2::GlobalReplace(str, *re.re_, re2Replacement);
}

} // namespace facebook::velox::regex_compat
