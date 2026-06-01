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
#include "velox/functions/lib/java_pcre2_translator/ClassBodyParser.h"
#include "velox/functions/lib/java_pcre2_translator/Evaluator.h"
#include "velox/functions/lib/java_pcre2_translator/JavaRegexTranslator.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <vector>

namespace facebook::velox::regex_compat {
namespace {

std::uint32_t toPcre2Options(const Options& o) {
  // Java's default \d, \s and \w shorthands are ASCII-only.  Keep UTF enabled
  // for Unicode literals and \p{...}, but do not enable PCRE2_UCP here.
  std::uint32_t opts = PCRE2_UTF;
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

void replaceAll(std::string& s, std::string_view from, std::string_view to) {
  for (std::size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos;
       pos += to.size()) {
    s.replace(pos, from.size(), to);
  }
}

std::string surrogateUtf8ByteEscapes(std::uint32_t cp) {
  char buf[32];
  std::snprintf(
      buf,
      sizeof(buf),
      "\\x{%02X}\\x{%02X}\\x{%02X}",
      0xE0 | (cp >> 12),
      0x80 | ((cp >> 6) & 0x3F),
      0x80 | (cp & 0x3F));
  return buf;
}

std::string rawSurrogateUtf8BytePattern(
    unsigned char b0,
    unsigned char b1,
    unsigned char b2) {
  char buf[40];
  std::snprintf(
      buf,
      sizeof(buf),
      "(?:\\x{%02X}\\x{%02X}\\x{%02X})",
      b0,
      b1,
      b2);
  return buf;
}

std::string byteEscape(unsigned char b) {
  char buf[8];
  std::snprintf(buf, sizeof(buf), "\\x{%02X}", b);
  return buf;
}

std::string codePointUtf8ByteEscapes(std::uint32_t cp) {
  if (cp <= 0x7F) {
    return byteEscape(static_cast<unsigned char>(cp));
  }
  if (cp <= 0x7FF) {
    return byteEscape(static_cast<unsigned char>(0xC0 | (cp >> 6))) +
        byteEscape(static_cast<unsigned char>(0x80 | (cp & 0x3F)));
  }
  if (cp <= 0xFFFF) {
    return surrogateUtf8ByteEscapes(cp);
  }
  return byteEscape(static_cast<unsigned char>(0xF0 | (cp >> 18))) +
      byteEscape(static_cast<unsigned char>(0x80 | ((cp >> 12) & 0x3F))) +
      byteEscape(static_cast<unsigned char>(0x80 | ((cp >> 6) & 0x3F))) +
      byteEscape(static_cast<unsigned char>(0x80 | (cp & 0x3F)));
}

std::uint64_t rangeSetSize(
    const functions::java_pcre2_translator::RangeSet& rs,
    std::uint64_t cap) {
  std::uint64_t size = 0;
  const auto& ranges = rs.ranges();
  for (std::size_t i = 0; i < ranges.size(); i += 2) {
    size += static_cast<std::uint64_t>(ranges[i + 1]) - ranges[i] + 1;
    if (size > cap) {
      return size;
    }
  }
  return size;
}

std::string enumerateCodePointSet(
    const functions::java_pcre2_translator::RangeSet& rs) {
  std::string out = "(?:";
  bool first = true;
  const auto& ranges = rs.ranges();
  for (std::size_t i = 0; i < ranges.size(); i += 2) {
    for (std::int32_t cp = ranges[i]; cp <= ranges[i + 1]; ++cp) {
      if (!first) {
        out.push_back('|');
      }
      out += codePointUtf8ByteEscapes(cp);
      first = false;
    }
  }
  out.push_back(')');
  return out;
}

std::string anyUtf8CodePointPattern() {
  return "(?:[\\x{00}-\\x{7F}]|"
      "[\\x{C2}-\\x{DF}][\\x{80}-\\x{BF}]|"
      "\\x{E0}[\\x{A0}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "[\\x{E1}-\\x{EC}\\x{EE}-\\x{EF}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "\\x{ED}[\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "\\x{F0}[\\x{90}-\\x{BF}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "[\\x{F1}-\\x{F3}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "\\x{F4}[\\x{80}-\\x{8F}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}])";
}

std::optional<std::string> utf8UpToPattern(std::int32_t maxCp) {
  if (maxCp >= functions::java_pcre2_translator::RangeSet::kMaxCp) {
    return anyUtf8CodePointPattern();
  }
  if (maxCp == 0x103FF) {
    return std::string("(?:[\\x{00}-\\x{7F}]|"
        "[\\x{C2}-\\x{DF}][\\x{80}-\\x{BF}]|"
        "\\x{E0}[\\x{A0}-\\x{BF}][\\x{80}-\\x{BF}]|"
        "[\\x{E1}-\\x{EC}\\x{EE}-\\x{EF}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
        "\\x{ED}[\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
        "\\x{F0}\\x{90}[\\x{80}-\\x{8F}][\\x{80}-\\x{BF}])");
  }
  return std::nullopt;
}

std::optional<std::string> renderRangeSetAsUtf8BytePattern(
    const functions::java_pcre2_translator::RangeSet& rs) {
  constexpr std::uint64_t kEnumerateLimit = 4096;
  if (rs.isEmpty()) {
    return std::string("(?!)");
  }
  if (rangeSetSize(rs, kEnumerateLimit) <= kEnumerateLimit) {
    return enumerateCodePointSet(rs);
  }

  auto excluded = functions::java_pcre2_translator::RangeSet::all()
                      .subtract(rs);
  auto anyPattern = anyUtf8CodePointPattern();
  const auto& ranges = rs.ranges();
  if (!ranges.empty() && ranges.front() == 0 &&
      ranges.back() < functions::java_pcre2_translator::RangeSet::kMaxCp) {
    const auto maxCp = ranges.back();
    excluded = functions::java_pcre2_translator::RangeSet::range(0, maxCp)
                   .subtract(rs);
    auto upTo = utf8UpToPattern(maxCp);
    if (!upTo.has_value()) {
      return std::nullopt;
    }
    anyPattern = *upTo;
  }
  if (rangeSetSize(excluded, 64) <= 64) {
    return std::string("(?!") + enumerateCodePointSet(excluded) + ")" +
        anyPattern;
  }
  return std::nullopt;
}

std::optional<std::string> tryRewriteClassAsUtf8BytePattern(
    std::string_view pattern,
    std::size_t start,
    std::size_t& end) {
  namespace translator = functions::java_pcre2_translator;
  try {
    std::size_t pos = start;
    const auto node = translator::ClassBodyParser::parseClass(pattern, pos);
    end = pos;
    const auto rs = translator::Evaluator::tryToRangeSet(node);
    if (!rs.has_value()) {
      return std::nullopt;
    }
    return renderRangeSetAsUtf8BytePattern(*rs);
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  }
}

bool rawSurrogateUtf8At(std::string_view s, std::size_t i) {
  if (i + 2 >= s.size()) {
    return false;
  }
  const auto b0 = static_cast<unsigned char>(s[i]);
  const auto b1 = static_cast<unsigned char>(s[i + 1]);
  const auto b2 = static_cast<unsigned char>(s[i + 2]);
  return b0 == 0xED && b1 >= 0xA0 && b1 <= 0xBF && b2 >= 0x80 &&
      b2 <= 0xBF;
}

bool containsRawSurrogateUtf8(std::string_view s) {
  for (std::size_t i = 0; i + 2 < s.size(); ++i) {
    if (rawSurrogateUtf8At(s, i)) {
      return true;
    }
  }
  return false;
}

std::string rewriteRawSurrogateUtf8Classes(std::string pattern) {
  std::string out;
  out.reserve(pattern.size());
  for (std::size_t i = 0; i < pattern.size();) {
    if (pattern[i] != '[') {
      out.push_back(pattern[i++]);
      continue;
    }

    const std::size_t start = i;
    std::size_t parsedEnd = i;
    if (auto rewritten =
            tryRewriteClassAsUtf8BytePattern(pattern, start, parsedEnd)) {
      const std::string_view classText(
          pattern.data() + start, parsedEnd - start);
      if (classText.find("&&") != std::string_view::npos ||
          containsRawSurrogateUtf8(classText)) {
        out += *rewritten;
        i = parsedEnd;
        continue;
      }
    }

    std::size_t j = i + 1;
    if (j < pattern.size() && pattern[j] == '^') {
      out.push_back(pattern[i++]);
      continue;
    }
    bool escaped = false;
    for (; j < pattern.size(); ++j) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (pattern[j] == '\\') {
        escaped = true;
        continue;
      }
      if (pattern[j] == ']') {
        break;
      }
    }
    if (j == pattern.size()) {
      out.push_back(pattern[i++]);
      continue;
    }

    const std::string_view body(pattern.data() + i + 1, j - i - 1);
    if (body.find("&&") != std::string_view::npos) {
      out.append(pattern, start, j + 1 - start);
      i = j + 1;
      continue;
    }

    std::string byteClass;
    std::vector<std::string> surrogateAlts;
    bool unsupportedRange = false;
    for (std::size_t k = 0; k < body.size();) {
      if (rawSurrogateUtf8At(body, k)) {
        if ((k > 0 && body[k - 1] == '-') ||
            (k + 3 < body.size() && body[k + 3] == '-')) {
          unsupportedRange = true;
          break;
        }
        surrogateAlts.push_back(rawSurrogateUtf8BytePattern(
            static_cast<unsigned char>(body[k]),
            static_cast<unsigned char>(body[k + 1]),
            static_cast<unsigned char>(body[k + 2])));
        k += 3;
        continue;
      }
      byteClass.push_back(body[k++]);
    }

    if (surrogateAlts.empty() || unsupportedRange) {
      out.append(pattern, start, j + 1 - start);
    } else {
      out += "(?:";
      bool needPipe = false;
      if (!byteClass.empty()) {
        out.push_back('[');
        out += byteClass;
        out.push_back(']');
        needPipe = true;
      }
      for (const auto& alt : surrogateAlts) {
        if (needPipe) {
          out.push_back('|');
        }
        out += alt;
        needPipe = true;
      }
      out.push_back(')');
    }
    i = j + 1;
  }
  return out;
}

std::string rewriteRawSurrogateUtf8Literals(std::string pattern) {
  std::string out;
  out.reserve(pattern.size());
  bool inClass = false;
  for (std::size_t i = 0; i < pattern.size();) {
    const char c = pattern[i];
    if (c == '\\' && i + 1 < pattern.size()) {
      out.push_back(pattern[i++]);
      out.push_back(pattern[i++]);
      continue;
    }
    if (c == '[') {
      inClass = true;
      out.push_back(c);
      ++i;
      continue;
    }
    if (c == ']' && inClass) {
      inClass = false;
      out.push_back(c);
      ++i;
      continue;
    }
    if (!inClass && rawSurrogateUtf8At(pattern, i)) {
      const auto b0 = static_cast<unsigned char>(pattern[i]);
      const auto b1 = static_cast<unsigned char>(pattern[i + 1]);
      const auto b2 = static_cast<unsigned char>(pattern[i + 2]);
      out += rawSurrogateUtf8BytePattern(b0, b1, b2);
      i += 3;
      continue;
    }
    out.push_back(c);
    ++i;
  }
  return out;
}

std::string rewriteSurrogateEscapesForRawByteMode(std::string pattern) {
  // The translator reports raw-byte mode via a side-channel bool.  PCRE2 in
  // non-UTF mode accepts literal surrogate UTF-8 bytes, but not \x{D800};
  // rewrite the surrogate block aliases to byte-sequence regexes before
  // dropping PCRE2_UTF.
  constexpr std::string_view kAnySurrogateBytes =
      "(?:\\x{ED}[\\x{A0}-\\x{AF}][\\x{80}-\\x{BF}]|"
      "\\x{ED}[\\x{B0}-\\x{BF}][\\x{80}-\\x{BF}])";
  constexpr std::string_view kLowSurrogateBytes =
      "(?:\\x{ED}[\\x{B0}-\\x{BF}][\\x{80}-\\x{BF}])";
  replaceAll(
      pattern,
      "[\\x{d800}-\\x{dbff}\\x{dc00}-\\x{dfff}]",
      kAnySurrogateBytes);
  replaceAll(
      pattern,
      "[\\x{D800}-\\x{DBFF}\\x{DC00}-\\x{DFFF}]",
      kAnySurrogateBytes);
  replaceAll(
      pattern,
      "[[\\x{D800}-\\x{DB7F}][\\x{DC00}-\\x{DFFF}]]",
      "(?:\\x{ED}[\\x{A0}-\\x{AD}][\\x{80}-\\x{BF}]|"
      "\\x{ED}\\x{AE}[\\x{80}-\\x{BF}]|"
      "\\x{ED}[\\x{B0}-\\x{BF}][\\x{80}-\\x{BF}])");
  replaceAll(
      pattern,
      "[[\\x{d800}-\\x{db7f}][\\x{dc00}-\\x{dfff}]]",
      "(?:\\x{ED}[\\x{A0}-\\x{AD}][\\x{80}-\\x{BF}]|"
      "\\x{ED}\\x{AE}[\\x{80}-\\x{BF}]|"
      "\\x{ED}[\\x{B0}-\\x{BF}][\\x{80}-\\x{BF}])");
  replaceAll(
      pattern,
      "[\\x{D800}-\\x{DB7F}\\x{DC00}-\\x{DFFF}]",
      "(?:\\x{ED}[\\x{A0}-\\x{AD}][\\x{80}-\\x{BF}]|"
      "\\x{ED}\\x{AE}[\\x{80}-\\x{BF}]|"
      "\\x{ED}[\\x{B0}-\\x{BF}][\\x{80}-\\x{BF}])");
  replaceAll(pattern, "[\\x{dc00}-\\x{dfff}]", kLowSurrogateBytes);
  replaceAll(pattern, "[\\x{DC00}-\\x{DFFF}]", kLowSurrogateBytes);
  replaceAll(
      pattern,
      "[\\x{D800}-\\x{DB7F}]",
      "(?:\\x{ED}[\\x{A0}-\\x{AD}][\\x{80}-\\x{BF}])");
  replaceAll(
      pattern,
      "[\\x{DB80}-\\x{DBFF}]",
      "(?:\\x{ED}[\\x{AE}-\\x{AF}][\\x{80}-\\x{BF}])");
  replaceAll(
      pattern,
      "[\\x{DC00}-\\x{DFFF}]",
      "(?:\\x{ED}[\\x{B0}-\\x{BF}][\\x{80}-\\x{BF}])");

  for (std::uint32_t cp = 0xD800; cp <= 0xDFFF; ++cp) {
    char token[16];
    std::snprintf(token, sizeof(token), "\\x{%04X}", cp);
    replaceAll(pattern, token, surrogateUtf8ByteEscapes(cp));
    std::snprintf(token, sizeof(token), "\\x{%04x}", cp);
    replaceAll(pattern, token, surrogateUtf8ByteEscapes(cp));
  }
  const std::string rawAnySurrogateRange =
      std::string("[") + std::string("\xED\xA0\x80", 3) + "-" +
      std::string("\xED\xBF\xBF", 3) + "]";
  const std::string rawNotAnySurrogateRange =
      std::string("[^") + std::string("\xED\xA0\x80", 3) + "-" +
      std::string("\xED\xBF\xBF", 3) + "]";
  constexpr std::string_view kValidUtf8NonSurrogate =
      "(?:[\\x{00}-\\x{7F}]|"
      "[\\x{C2}-\\x{DF}][\\x{80}-\\x{BF}]|"
      "\\x{E0}[\\x{A0}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "[\\x{E1}-\\x{EC}\\x{EE}-\\x{EF}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "\\x{ED}[\\x{80}-\\x{9F}][\\x{80}-\\x{BF}]|"
      "\\x{F0}[\\x{90}-\\x{BF}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "[\\x{F1}-\\x{F3}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}]|"
      "\\x{F4}[\\x{80}-\\x{8F}][\\x{80}-\\x{BF}][\\x{80}-\\x{BF}])";
  replaceAll(pattern, rawNotAnySurrogateRange, kValidUtf8NonSurrogate);
  replaceAll(
      pattern,
      rawAnySurrogateRange,
      "(?:\\x{ED}[\\x{A0}-\\x{BF}][\\x{80}-\\x{BF}])");
  return rewriteRawSurrogateUtf8Literals(
      rewriteRawSurrogateUtf8Classes(std::move(pattern)));
}

bool containsSurrogateUtf8(std::string_view s) {
  for (std::size_t i = 0; i + 2 < s.size(); ++i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const auto b1 = static_cast<unsigned char>(s[i + 1]);
    const auto b2 = static_cast<unsigned char>(s[i + 2]);
    if (b0 == 0xED && b1 >= 0xA0 && b1 <= 0xBF && b2 >= 0x80 &&
        b2 <= 0xBF) {
      return true;
    }
  }
  return false;
}

} // namespace

Pcre2Regex::Pcre2Regex(std::string_view javaPattern, Options opt) {
  // Translate Java regex syntax → PCRE2 syntax before compiling.  When
  // the translator cannot express the pattern in PCRE2 (e.g. an
  // unsupported `\p{...}` property in an intersection), we report the
  // translator message verbatim and leave the pattern uncompiled.
  std::string pcre2Pattern;
  bool needsRawByteMode = false;
  try {
    pcre2Pattern = functions::java_pcre2_translator::toPcre2Pattern(
        javaPattern, needsRawByteMode);
  } catch (const functions::java_pcre2_translator::EvaluationFailedException&
               ex) {
    error_ = std::string("Java→PCRE2 translator: ") + ex.what();
    return;
  }
  if (needsRawByteMode) {
    pcre2Pattern = rewriteSurrogateEscapesForRawByteMode(std::move(pcre2Pattern));
  }

  int err = 0;
  PCRE2_SIZE off = 0;
  code_ = pcre2_compile_8(
      reinterpret_cast<PCRE2_SPTR8>(pcre2Pattern.data()),
      pcre2Pattern.size(),
      toPcre2Options(opt) & (needsRawByteMode ? ~PCRE2_UTF : ~0u),
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
      toPcre2MatchOptions(anchor) |
          (containsSurrogateUtf8(input.substr(0, endpos)) ? PCRE2_NO_UTF_CHECK
                                                          : 0),
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
