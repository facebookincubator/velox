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
#include "velox/external/regex_compat/Pcre2Regex.h"
#include "velox/functions/lib/java_pcre2_translator/JavaRegexTranslator.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

namespace facebook::velox::regex_compat {
namespace {

std::uint32_t toPcre2Options(const Options& o) {
  // PCRE2_UTF + PCRE2_UCP: UTF-8 input + Unicode-aware character properties.
  std::uint32_t opts = PCRE2_UTF | PCRE2_UCP;
  if (!o.caseSensitive) {
    opts |= PCRE2_CASELESS;
  }
  if (o.dotNl) {
    opts |= PCRE2_DOTALL;
  }
  if (!o.oneLine) {
    opts |= PCRE2_MULTILINE;
  }
  return opts;
}

std::uint32_t toPcre2MatchOptions(Anchor a) {
  switch (a) {
    case Anchor::kUnanchored:
      return 0;
    case Anchor::kAnchorStart:
      return PCRE2_ANCHORED;
    case Anchor::kAnchorBoth:
      return PCRE2_ANCHORED | PCRE2_ENDANCHORED;
  }
  return 0;
}

std::string pcre2ErrorToString(int code, PCRE2_SIZE offset) {
  PCRE2_UCHAR buf[256];
  pcre2_get_error_message(code, buf, sizeof(buf));
  std::ostringstream os;
  os << "PCRE2 error " << code << " at offset " << offset << ": "
     << reinterpret_cast<const char*>(buf);
  return os.str();
}

} // namespace

Pcre2Regex::Pcre2Regex(std::string_view javaPattern, Options opt) {
  // Translate Java regex syntax → PCRE2 syntax before compiling.  When
  // the translator cannot express the pattern in PCRE2 (e.g. an
  // unsupported `\p{...}` property in an intersection), we report the
  // translator message verbatim and leave the pattern uncompiled.
  std::string pcre2Pattern;
  try {
    pcre2Pattern = functions::java_pcre2_translator::toPcre2Pattern(javaPattern);
  } catch (const functions::java_pcre2_translator::EvaluationFailedException&
               ex) {
    error_ = std::string("Java→PCRE2 translator: ") + ex.what();
    return;
  }

  int err = 0;
  PCRE2_SIZE off = 0;
  code_ = pcre2_compile_8(
      reinterpret_cast<PCRE2_SPTR8>(pcre2Pattern.data()),
      pcre2Pattern.size(),
      toPcre2Options(opt),
      &err,
      &off,
      nullptr);
  if (!code_) {
    error_ = pcre2ErrorToString(err, off);
    return;
  }
  // JIT-compile for speed.  Falls back to the interpreter on platforms where
  // JIT isn't supported, no special handling needed.
  pcre2_jit_compile_8(code_, PCRE2_JIT_COMPLETE);

  // Capture count.
  std::uint32_t cap = 0;
  pcre2_pattern_info_8(code_, PCRE2_INFO_CAPTURECOUNT, &cap);
  captureCount_ = static_cast<int>(cap);

  // Named groups: name table is a flat blob of fixed-size entries; first 2
  // bytes of each entry are the (big-endian) group index, then a NUL-terminated
  // name.
  std::uint32_t nameCount = 0;
  std::uint32_t entrySize = 0;
  PCRE2_SPTR8 nameTable = nullptr;
  pcre2_pattern_info_8(code_, PCRE2_INFO_NAMECOUNT, &nameCount);
  pcre2_pattern_info_8(code_, PCRE2_INFO_NAMEENTRYSIZE, &entrySize);
  pcre2_pattern_info_8(code_, PCRE2_INFO_NAMETABLE, &nameTable);
  for (std::uint32_t i = 0; i < nameCount; ++i) {
    const std::uint8_t* entry = nameTable + i * entrySize;
    int idx = (entry[0] << 8) | entry[1];
    named_.emplace(reinterpret_cast<const char*>(entry + 2), idx);
  }
}

Pcre2Regex::~Pcre2Regex() {
  if (code_) {
    pcre2_code_free_8(code_);
  }
}

bool Pcre2Regex::ok() const {
  return code_ != nullptr;
}
const std::string& Pcre2Regex::error() const {
  return error_;
}
int Pcre2Regex::NumberOfCapturingGroups() const {
  return captureCount_;
}
const std::map<std::string, int>& Pcre2Regex::NamedCapturingGroups() const {
  return named_;
}

bool Pcre2Regex::Match(
    std::string_view input,
    std::size_t startpos,
    std::size_t endpos,
    Anchor anchor,
    std::string_view* submatch,
    int nsubmatch) const {
  if (!code_) {
    return false;
  }
  pcre2_match_data_8* md =
      pcre2_match_data_create_from_pattern_8(code_, nullptr);
  // PCRE2 takes the full subject + the length to consider; passing `endpos`
  // as the length cleanly caps matching to [startpos, endpos).
  int rc = pcre2_match_8(
      code_,
      reinterpret_cast<PCRE2_SPTR8>(input.data()),
      endpos,
      startpos,
      toPcre2MatchOptions(anchor),
      md,
      nullptr);
  if (rc < 0) {
    pcre2_match_data_free_8(md);
    return false;
  }
  PCRE2_SIZE* ov = pcre2_get_ovector_pointer_8(md);
  int avail = std::min<int>(nsubmatch, rc);
  for (int i = 0; i < avail; ++i) {
    if (ov[2 * i] == PCRE2_UNSET) {
      submatch[i] = std::string_view{};
    } else {
      submatch[i] = input.substr(ov[2 * i], ov[2 * i + 1] - ov[2 * i]);
    }
  }
  for (int i = avail; i < nsubmatch; ++i) {
    submatch[i] = std::string_view{};
  }
  pcre2_match_data_free_8(md);
  return true;
}

bool Pcre2Regex::FullMatch(std::string_view input, const Pcre2Regex& re) {
  std::string_view sub[1];
  return re.Match(input, 0, input.size(), Anchor::kAnchorBoth, sub, 1);
}

bool Pcre2Regex::PartialMatch(std::string_view input, const Pcre2Regex& re) {
  std::string_view sub[1];
  return re.Match(input, 0, input.size(), Anchor::kUnanchored, sub, 1);
}

int Pcre2Regex::GlobalReplace(
    std::string* str,
    const Pcre2Regex& re,
    std::string_view javaReplacement) {
  if (!re.ok() || str == nullptr) {
    return 0;
  }
  // PCRE2_SUBSTITUTE_EXTENDED enables $N / ${name} / $$ / \$ — the Java
  // replacement syntax that Velox's `prepareRegexpReplaceReplacement` had to
  // translate away for RE2.
  std::uint32_t opts = PCRE2_SUBSTITUTE_GLOBAL | PCRE2_SUBSTITUTE_EXTENDED |
      PCRE2_SUBSTITUTE_OVERFLOW_LENGTH;
  // First try with a reasonable initial buffer; on overflow PCRE2 tells us
  // the required size in `outlen` and we retry.
  std::string out;
  out.resize(str->size() * 2 + 32);
  PCRE2_SIZE outlen = out.size();
  int rc = pcre2_substitute_8(
      re.code_,
      reinterpret_cast<PCRE2_SPTR8>(str->data()),
      str->size(),
      0,
      opts,
      nullptr,
      nullptr,
      reinterpret_cast<PCRE2_SPTR8>(javaReplacement.data()),
      javaReplacement.size(),
      reinterpret_cast<PCRE2_UCHAR8*>(out.data()),
      &outlen);
  if (rc == PCRE2_ERROR_NOMEMORY) {
    out.resize(outlen);
    outlen = out.size();
    rc = pcre2_substitute_8(
        re.code_,
        reinterpret_cast<PCRE2_SPTR8>(str->data()),
        str->size(),
        0,
        opts,
        nullptr,
        nullptr,
        reinterpret_cast<PCRE2_SPTR8>(javaReplacement.data()),
        javaReplacement.size(),
        reinterpret_cast<PCRE2_UCHAR8*>(out.data()),
        &outlen);
  }
  if (rc < 0) {
    // Substitution error (e.g. unknown group); leave *str untouched.
    return 0;
  }
  out.resize(outlen);
  *str = std::move(out);
  return rc;
}

} // namespace facebook::velox::regex_compat
