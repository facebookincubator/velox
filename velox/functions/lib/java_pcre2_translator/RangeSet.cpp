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
//
// Originally authored by Oleksii PELYKH for pcre4j; ported from
// org.pcre4j.regex.translate.RangeSet (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/RangeSet.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace facebook::velox::functions::java_pcre2_translator {

namespace {

// Emit a single code point inside a PCRE2 character class body — mirrors
// `ClassRenderer.emitLiteralInClass` from the Java sources.  We inline
// it here to avoid a circular dep on the (yet to be ported) ClassRenderer
// module.  When Phase 4 lands `ClassRenderer`, we can either keep this
// helper local to RangeSet or expose it; the function bodies are
// trivial enough that duplication is fine.
void emitLiteralInClass(std::int32_t cp, std::string& sb) {
  if (cp >= 0x20 && cp <= 0x7E) {
    switch (cp) {
      case '\\':
      case ']':
      case '^':
      case '-':
        sb.push_back('\\');
        sb.push_back(static_cast<char>(cp));
        return;
      default:
        sb.push_back(static_cast<char>(cp));
        return;
    }
  }
  char buf[16];
  std::snprintf(buf, sizeof(buf), "\\x{%X}", static_cast<unsigned>(cp));
  sb.append(buf);
}

} // namespace

const RangeSet& RangeSet::empty() {
  static const RangeSet kEmpty{{}};
  return kEmpty;
}

const RangeSet& RangeSet::all() {
  static const RangeSet kAll{{0, kMaxCp}};
  return kAll;
}

RangeSet RangeSet::single(std::int32_t cp) {
  if (cp < 0 || cp > kMaxCp) {
    throw std::invalid_argument(
        "Code point out of range: " + std::to_string(cp));
  }
  return RangeSet({cp, cp});
}

RangeSet RangeSet::range(std::int32_t lo, std::int32_t hi) {
  if (lo < 0 || hi > kMaxCp || lo > hi) {
    throw std::invalid_argument(
        "Invalid range: [" + std::to_string(lo) + ", " + std::to_string(hi) +
        "]");
  }
  return RangeSet({lo, hi});
}

RangeSet RangeSet::unionWith(const RangeSet& other) const {
  if (isEmpty()) {
    return other;
  }
  if (other.isEmpty()) {
    return *this;
  }
  const auto& a = ranges_;
  const auto& b = other.ranges_;
  std::vector<std::int32_t> merged;
  merged.reserve(a.size() + b.size());
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      merged.push_back(a[i]);
      merged.push_back(a[i + 1]);
      i += 2;
    } else {
      merged.push_back(b[j]);
      merged.push_back(b[j + 1]);
      j += 2;
    }
  }
  while (i < a.size()) {
    merged.push_back(a[i]);
    merged.push_back(a[i + 1]);
    i += 2;
  }
  while (j < b.size()) {
    merged.push_back(b[j]);
    merged.push_back(b[j + 1]);
    j += 2;
  }
  return normalise(std::move(merged));
}

RangeSet RangeSet::intersect(const RangeSet& other) const {
  if (isEmpty() || other.isEmpty()) {
    return empty();
  }
  const auto& a = ranges_;
  const auto& b = other.ranges_;
  std::vector<std::int32_t> out;
  out.reserve(std::min(a.size(), b.size()));
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    const std::int32_t lo = std::max(a[i], b[j]);
    const std::int32_t hi = std::min(a[i + 1], b[j + 1]);
    if (lo <= hi) {
      out.push_back(lo);
      out.push_back(hi);
    }
    if (a[i + 1] < b[j + 1]) {
      i += 2;
    } else {
      j += 2;
    }
  }
  if (out.empty()) {
    return empty();
  }
  return RangeSet(std::move(out));
}

RangeSet RangeSet::complement() const {
  if (isEmpty()) {
    return all();
  }
  std::vector<std::int32_t> out;
  out.reserve(ranges_.size() + 2);
  std::int32_t prev = 0;
  for (std::size_t i = 0; i < ranges_.size(); i += 2) {
    if (prev < ranges_[i]) {
      out.push_back(prev);
      out.push_back(ranges_[i] - 1);
    }
    prev = ranges_[i + 1] + 1;
  }
  if (prev <= kMaxCp) {
    out.push_back(prev);
    out.push_back(kMaxCp);
  }
  if (out.empty()) {
    return empty();
  }
  return RangeSet(std::move(out));
}

RangeSet RangeSet::subtract(const RangeSet& other) const {
  return intersect(other.complement());
}

bool RangeSet::contains(std::int32_t cp) const {
  for (std::size_t i = 0; i < ranges_.size(); i += 2) {
    if (cp >= ranges_[i] && cp <= ranges_[i + 1]) {
      return true;
    }
    if (cp < ranges_[i]) {
      return false;
    }
  }
  return false;
}

std::string RangeSet::toPcre2ClassBody() const {
  std::string sb;
  for (std::size_t i = 0; i < ranges_.size(); i += 2) {
    const std::int32_t lo = ranges_[i];
    const std::int32_t hi = ranges_[i + 1];
    emitLiteralInClass(lo, sb);
    if (lo != hi) {
      sb.push_back('-');
      emitLiteralInClass(hi, sb);
    }
  }
  return sb;
}

RangeSet RangeSet::normalise(std::vector<std::int32_t>&& raw) {
  if (raw.empty()) {
    return empty();
  }
  std::vector<std::int32_t> out;
  out.reserve(raw.size());
  std::int32_t curLo = raw[0];
  std::int32_t curHi = raw[1];
  for (std::size_t i = 2; i < raw.size(); i += 2) {
    const std::int32_t lo = raw[i];
    const std::int32_t hi = raw[i + 1];
    if (lo <= curHi + 1) {
      curHi = std::max(curHi, hi);
    } else {
      out.push_back(curLo);
      out.push_back(curHi);
      curLo = lo;
      curHi = hi;
    }
  }
  out.push_back(curLo);
  out.push_back(curHi);
  return RangeSet(std::move(out));
}

} // namespace facebook::velox::functions::java_pcre2_translator
