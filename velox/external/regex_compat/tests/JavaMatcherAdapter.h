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
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "velox/external/regex_compat/RegexTypes.h"

namespace facebook::velox::regex_compat::test {

/// Header-only adapter that reconstructs Java `java.util.regex.Matcher`'s
/// stateful API (`find()` cursor, `group(int)`, `start/end`, `replaceAll`,
/// etc.) on top of the stateless `IRegex`-shaped `Match()` method exposed
/// by the three backend classes.
///
/// Lives in the test target only — the production backend classes
/// deliberately do not carry this Matcher state, to keep their surface
/// close to `re2::RE2`'s actual usage in Velox.
template <typename R>
class JavaMatcherAdapter {
 public:
  JavaMatcherAdapter(const R* re, std::string_view input)
      : re_(re),
        input_(input),
        regionStart_(0),
        regionEnd_(input.size()),
        // +1 for group 0 (full match).
        groups_(re->NumberOfCapturingGroups() + 1) {}

  // ----- find()/matches() family -----

  /// Advance past the previous match and search forward.
  bool find() {
    if (!re_->ok()) {
      matched_ = false;
      return false;
    }
    if (cursor_ > regionEnd_) {
      matched_ = false;
      return false;
    }
    matched_ = re_->Match(
        input_,
        cursor_,
        regionEnd_,
        Anchor::kUnanchored,
        groups_.data(),
        static_cast<int>(groups_.size()));
    if (!matched_) {
      return false;
    }
    const std::size_t s = matchBeg();
    const std::size_t e = matchEnd();
    // Zero-width match: advance by 1 to avoid an infinite loop, mirroring
    // java.util.regex.Matcher semantics.
    cursor_ = (s == e) ? e + 1 : e;
    return true;
  }

  /// Reset cursor to `start`, then `find()` once.
  bool find(int start) {
    cursor_ = static_cast<std::size_t>(start);
    return find();
  }

  /// Anchored full-input match (Java `Matcher.matches`).  Does not advance
  /// the find-cursor.  Honors the active region.
  bool matches() {
    matched_ = re_->Match(
        input_,
        regionStart_,
        regionEnd_,
        Anchor::kAnchorBoth,
        groups_.data(),
        static_cast<int>(groups_.size()));
    return matched_;
  }

  /// Anchored prefix match (Java `Matcher.lookingAt`).  Honors the active
  /// region.
  bool lookingAt() {
    matched_ = re_->Match(
        input_,
        regionStart_,
        regionEnd_,
        Anchor::kAnchorStart,
        groups_.data(),
        static_cast<int>(groups_.size()));
    return matched_;
  }

  void reset() {
    cursor_ = regionStart_;
    matched_ = false;
  }

  void reset(std::string_view input) {
    input_ = input;
    regionStart_ = 0;
    regionEnd_ = input.size();
    reset();
  }

  /// Java `Matcher.region(start, end)` — restrict matching to a sub-range.
  /// Returns *this for chainability (matches Java's fluent API).  Also
  /// resets the find() cursor to `start`.
  JavaMatcherAdapter& region(int start, int end) {
    regionStart_ = static_cast<std::size_t>(start);
    regionEnd_ = static_cast<std::size_t>(end);
    cursor_ = regionStart_;
    matched_ = false;
    return *this;
  }

  // ----- Group accessors -----

  int groupCount() const {
    return re_->NumberOfCapturingGroups();
  }

  /// `Matcher.group(i)` — returns the captured substring for group `i`
  /// (0-based whole match = group 0).  Returns `std::nullopt` if the group
  /// did not participate in the last match.
  std::optional<std::string_view> group(int i) const {
    requireMatched();
    if (i < 0 || i >= static_cast<int>(groups_.size())) {
      throw std::out_of_range("group index out of range");
    }
    if (groups_[i].data() == nullptr) {
      return std::nullopt;
    }
    return groups_[i];
  }

  std::optional<std::string_view> group(const std::string& name) const {
    requireMatched();
    const auto& named = re_->NamedCapturingGroups();
    auto it = named.find(name);
    if (it == named.end()) {
      throw std::out_of_range("unknown group name: " + name);
    }
    return group(it->second);
  }

  int start(int i = 0) const {
    requireMatched();
    if (i < 0 || i >= static_cast<int>(groups_.size())) {
      throw std::out_of_range("group index out of range");
    }
    if (groups_[i].data() == nullptr) {
      return -1;
    }
    return static_cast<int>(groups_[i].data() - input_.data());
  }

  int end(int i = 0) const {
    requireMatched();
    if (i < 0 || i >= static_cast<int>(groups_.size())) {
      throw std::out_of_range("group index out of range");
    }
    if (groups_[i].data() == nullptr) {
      return -1;
    }
    return static_cast<int>(
        groups_[i].data() + groups_[i].size() - input_.data());
  }

  int start(const std::string& name) const {
    const auto& named = re_->NamedCapturingGroups();
    auto it = named.find(name);
    if (it == named.end()) {
      throw std::out_of_range("unknown group name: " + name);
    }
    return start(it->second);
  }

  int end(const std::string& name) const {
    const auto& named = re_->NamedCapturingGroups();
    auto it = named.find(name);
    if (it == named.end()) {
      throw std::out_of_range("unknown group name: " + name);
    }
    return end(it->second);
  }

  // ----- Replacement -----

  /// `Matcher.replaceAll(repl)`: delegates to backend's GlobalReplace.  The
  /// replacement string uses Java `\$N` / `\${name}` syntax.
  std::string replaceAll(std::string_view javaReplacement) const {
    std::string s(input_);
    R::GlobalReplace(&s, *re_, javaReplacement);
    return s;
  }

  /// `Matcher.replaceFirst(repl)`: replace only the first match.  We do this
  /// by walking find() once, building the result manually.
  std::string replaceFirst(std::string_view javaReplacement) {
    JavaMatcherAdapter copy(re_, input_);
    if (!copy.find()) {
      return std::string(input_);
    }
    // Build by hand using backend's GlobalReplace on a one-match window:
    // easiest correctness path is to call GlobalReplace on a string that
    // contains only the first match in-place — but that's awkward.
    // Instead, recompose: prefix + expand(repl, groups) + suffix.
    const std::size_t s = copy.matchBeg();
    const std::size_t e = copy.matchEnd();
    std::string out;
    out.reserve(input_.size() + javaReplacement.size());
    out.append(input_.substr(0, s));
    out.append(expandJavaReplacement(javaReplacement, copy.groups_));
    out.append(input_.substr(e));
    return out;
  }

  /// `Matcher.appendReplacement(sb, repl)`: stateful incremental replace.
  /// Appends to `sb` the prefix-since-last-call plus the expanded
  /// replacement for the most recent match.  Must be called only after a
  /// successful `find()`.  Throws `std::logic_error` (mirrors Java's
  /// `IllegalStateException`) if no match is available.
  void appendReplacement(std::string& sb, std::string_view javaReplacement) {
    if (!matched_) {
      throw std::logic_error(
          "appendReplacement: no match available (call find() first)");
    }
    const std::size_t s = matchBeg();
    const std::size_t e = matchEnd();
    sb.append(input_.substr(lastAppendPos_, s - lastAppendPos_));
    sb.append(expandJavaReplacement(javaReplacement, groups_));
    lastAppendPos_ = e;
  }

  /// `Matcher.appendTail(sb)`: appends input from lastAppendPosition to end.
  void appendTail(std::string& sb) const {
    sb.append(input_.substr(lastAppendPos_));
  }

  /// `Matcher.quoteReplacement(s)` static: escape `$` and `\` in `s` so it
  /// can be safely used as a literal replacement.
  static std::string quoteReplacement(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      if (c == '\\' || c == '$') {
        out.push_back('\\');
      }
      out.push_back(c);
    }
    return out;
  }

 private:
  std::size_t matchBeg() const {
    return groups_[0].data() - input_.data();
  }
  std::size_t matchEnd() const {
    return matchBeg() + groups_[0].size();
  }
  void requireMatched() const {
    if (!matched_) {
      throw std::logic_error("no match available");
    }
  }

  // Expand Java replacement string ($N / ${name} / \\$ / \\\\) using the
  // given group slots.  Public-style helper used by replaceFirst.  We don't
  // route through R::GlobalReplace here because that re-matches the whole
  // input — we already have the groups in hand.
  std::string expandJavaReplacement(
      std::string_view r,
      const std::vector<std::string_view>& g) const {
    std::string out;
    out.reserve(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) {
      char c = r[i];
      if (c == '\\' && i + 1 < r.size()) {
        out.push_back(r[i + 1]);
        ++i;
      } else if (c == '$' && i + 1 < r.size()) {
        char n = r[i + 1];
        if (n >= '0' && n <= '9') {
          int idx = n - '0';
          if (idx < static_cast<int>(g.size()) && g[idx].data() != nullptr) {
            out.append(g[idx]);
          }
          ++i;
        } else if (n == '{') {
          auto endBrace = r.find('}', i + 2);
          if (endBrace == std::string_view::npos) {
            out.push_back(c);
            continue;
          }
          const std::string name(r.substr(i + 2, endBrace - i - 2));
          const auto& named = re_->NamedCapturingGroups();
          auto it = named.find(name);
          if (it != named.end() && it->second < static_cast<int>(g.size()) &&
              g[it->second].data() != nullptr) {
            out.append(g[it->second]);
          }
          i = endBrace;
        } else {
          out.push_back(c);
        }
      } else {
        out.push_back(c);
      }
    }
    return out;
  }

  const R* re_;
  std::string_view input_;
  std::size_t regionStart_ = 0;
  std::size_t regionEnd_ = 0;
  std::size_t cursor_ = 0;
  std::size_t lastAppendPos_ = 0;
  bool matched_ = false;
  std::vector<std::string_view> groups_;
};

} // namespace facebook::velox::regex_compat::test
