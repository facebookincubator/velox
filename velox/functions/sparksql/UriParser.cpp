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

#include "velox/functions/sparksql/UriParser.h"

#include <array>
#include <cstdint>
#include <limits>

#include "velox/common/base/SimdUtil.h"

#if XSIMD_WITH_AVX2
#include <immintrin.h>
#endif

#include "folly/CPortability.h"
#include "velox/external/utf8proc/utf8procImpl.h"

namespace facebook::velox::functions::sparksql {

namespace {
// Character classes that define the URL grammar. Each predicate answers
// "may this character appear here" for one component: the generic sets
// (unreserved, reserved) build up the per-component sets (path, query,
// userinfo, authority). Predicates over ASCII characters take char and
// are evaluated per byte; the two code-point predicates (isSpaceChar,
// isIsoControl) take the int32_t produced by utf8proc_codepoint and
// classify decoded non-ASCII characters.

// ASCII letters. A scheme and a hostname label must both start with one.
FOLLY_ALWAYS_INLINE constexpr bool isAlpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// ASCII decimal digits: the alphabet of port numbers and IPv4 bytes.
FOLLY_ALWAYS_INLINE constexpr bool isDigit(char c) {
  return c >= '0' && c <= '9';
}

// ASCII letters or digits.
FOLLY_ALWAYS_INLINE constexpr bool isAlnum(char c) {
  return isAlpha(c) || isDigit(c);
}

// Hexadecimal digits: the payload of a "%XX" escape pair.
FOLLY_ALWAYS_INLINE constexpr bool isHexDigit(char c) {
  return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

// Punctuation that carries no structural meaning in a URL, so it can
// appear in any component without being escaped.
FOLLY_ALWAYS_INLINE constexpr bool isMark(char c) {
  switch (c) {
    case '-':
    case '_':
    case '.':
    case '!':
    case '~':
    case '*':
    case '\'':
    case '(':
    case ')':
      return true;
    default:
      return false;
  }
}

// Characters with no syntactic role in a URL: alphanumerics plus the
// marks above. Accepted by every component grammar in this file.
FOLLY_ALWAYS_INLINE constexpr bool isUnreserved(char c) {
  return isAlnum(c) || isMark(c);
}

// Delimiters that structure a URL (separating components or query
// parameters). They are allowed inside queries and fragments, where
// they carry no meaning, but mark boundaries everywhere else; brackets
// are included because they may appear unescaped there too.
FOLLY_ALWAYS_INLINE constexpr bool isReserved(char c) {
  switch (c) {
    case ';':
    case '/':
    case '?':
    case ':':
    case '@':
    case '&':
    case '=':
    case '+':
    case '$':
    case ',':
    case '[':
    case ']':
      return true;
    default:
      return false;
  }
}

// Punctuation that may appear inside a path segment beyond the
// unreserved set.
FOLLY_ALWAYS_INLINE constexpr bool isPcharExtra(char c) {
  switch (c) {
    case ':':
    case '@':
    case '&':
    case '=':
    case '+':
    case '$':
    case ',':
      return true;
    default:
      return false;
  }
}

// Everything a query, fragment, or opaque scheme-specific part may
// contain: the unreserved characters plus the reserved delimiters,
// which carry no meaning inside those components.
FOLLY_ALWAYS_INLINE constexpr bool isUric(char c) {
  return isUnreserved(c) || isReserved(c);
}

// The path grammar: unreserved characters, segment punctuation, and
// the '/' that separates segments.
FOLLY_ALWAYS_INLINE constexpr bool isPathChar(char c) {
  return isUnreserved(c) || isPcharExtra(c) || c == ';' || c == '/';
}

// The "user:password" grammar of an authority, the part before '@'.
// It excludes '@' itself, which ends the userinfo and starts the host.
FOLLY_ALWAYS_INLINE constexpr bool isUserInfoChar(char c) {
  return isUnreserved(c) || c == ';' || c == ':' || c == '&' || c == '=' ||
      c == '+' || c == '$' || c == ',';
}

// The authority grammar for a non-server authority (one without a
// recognizable host): free-form text, tolerated but never decomposed.
FOLLY_ALWAYS_INLINE constexpr bool isRegNameChar(char c) {
  return isUnreserved(c) || isPcharExtra(c) || c == ';' || c == '$';
}

// The authority grammar for a server-based authority ("host[:port]"
// with optional userinfo and bracketed IPv6 literals), excluding '%'.
FOLLY_ALWAYS_INLINE constexpr bool isServerChar(char c) {
  return isUnreserved(c) || isPcharExtra(c) || c == ';' || c == '[' || c == ']';
}

// The server authority set plus '%', which appears in IPv6 scope ids
// ("[fe80::1%eth0]"); '%' is a mask member rather than an escape start
// in that position.
FOLLY_ALWAYS_INLINE constexpr bool isServerPercentChar(char c) {
  return isServerChar(c) || c == '%';
}

// Characters of a scheme name past its first letter, which must be a
// letter: alphanumerics plus '+', '-' and '.'.
FOLLY_ALWAYS_INLINE constexpr bool isSchemeChar(char c) {
  return isAlnum(c) || c == '+' || c == '-' || c == '.';
}

// A label of a hostname, past its first character: alphanumerics and
// dashes (a trailing dash is rejected by parseHostname itself).
FOLLY_ALWAYS_INLINE constexpr bool isAlnumOrDash(char c) {
  return isAlnum(c) || c == '-';
}

// A digit or a dot, the characters an IPv4 address may contain.
FOLLY_ALWAYS_INLINE constexpr bool isIpv4Char(char c) {
  return isDigit(c) || c == '.';
}

// A hexadecimal digit, the only character an IPv6 group may contain.
FOLLY_ALWAYS_INLINE constexpr bool isHexGroupChar(char c) {
  return isHexDigit(c);
}

// The characters an IPv6 scope id may contain: alphanumerics plus '_'
// and '.'.
FOLLY_ALWAYS_INLINE constexpr bool isScopeIdChar(char c) {
  return isAlnum(c) || c == '_' || c == '.';
}

// Returns the leading run of characters accepted by Pred, without
// consuming input. The bounded-loop spelling of find_first_not_of with
// a literal set, so the accepted set is stated by a predicate's name.
template <bool (*Pred)(char)>
FOLLY_ALWAYS_INLINE std::string_view leadingRun(std::string_view input) {
  size_t length = 0;
  while (length < input.size() && Pred(input[length])) {
    ++length;
  }
  return input.substr(0, length);
}

// The leading run of hex digits (an IPv6 group prefix).
FOLLY_ALWAYS_INLINE std::string_view leadingHexDigits(std::string_view input) {
  return leadingRun<isHexGroupChar>(input);
}

// ASCII and Unicode control characters (C0, plus DEL and the C1 range):
// never acceptable in a URL, not even escaped.
FOLLY_ALWAYS_INLINE constexpr bool isIsoControl(utf8proc_int32_t codePoint) {
  return codePoint <= 0x1F || (codePoint >= 0x7F && codePoint <= 0x9F);
}

// Separator characters (the Unicode Zs, Zl, and Zp categories, which
// includes plain space): never acceptable in a URL. Together with the
// control characters above, these are the only non-ASCII code points a
// URL may not contain.
FOLLY_ALWAYS_INLINE constexpr bool isSpaceChar(utf8proc_int32_t codePoint) {
  switch (codePoint) {
    case 0x20:
    case 0xA0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
      return true;
    default:
      return codePoint >= 0x2000 && codePoint <= 0x200A;
  }
}

// A predicate's ASCII membership packed as 16 bytes: bit (c & 7) of byte
// (c >> 3) is set iff Pred(c). skipAccepted feeds this table to the
// vectorized lookup.
template <bool (*Pred)(char)>
struct AsciiBitmap {
  static constexpr std::array<uint8_t, 16> kValues = [] {
    std::array<uint8_t, 16> values{};
    for (int c = 0; c < 128; ++c) {
      if (Pred(static_cast<char>(c))) {
        values[c >> 3] |= static_cast<uint8_t>(1 << (c & 7));
      }
    }
    return values;
  }();
};

// Why a run of accepted characters ended.
enum class ScanStop {
  kEnd, //< The whole input was accepted.
  kRejected, //< A character Pred rejects stopped the scan.
  kEscape, //< A '%' that may start an escape pair stopped the scan.
  kMultibyte, //< A byte >= 0x80 stopped the scan.
};

// Advances input past its leading characters accepted by Pred and
// reports why the scan stopped. Bytes >= 0x80 always stop the bulk scan
// because multi-byte characters need UTF-8 decoding, which is
// scanComponent's job. '%' stops the scan only when Pred rejects it
// (isServerPercentChar accepts it, for IPv6 scope ids); when it does
// stop there, it is reported as kEscape so scanComponent can validate
// the pair. Runs of accepted bytes are skipped 32 at a time with a
// two-shuffle table lookup (byte -> bitmap byte -> bit); the same
// bitmap drives the scalar tail, so new character sets only need a
// predicate - never new SIMD code.
template <bool (*Pred)(char)>
FOLLY_ALWAYS_INLINE ScanStop skipAccepted(std::string_view& input) {
  size_t i = 0;
  const size_t size = input.size();
// Because no portable SIMD library exposes a data-driven byte shuffle, so we
// need to fall back on a scalar implementation.
#if XSIMD_WITH_AVX2
  for (; i + 32 <= size; i += 32) {
    // Bulk-scan one 32-byte chunk per iteration. Each byte is classified
    // by two chained pshufb table lookups, the vectorized spelling of the
    // scalar test `bitmap[c >> 3] & (1 << (c & 7))`:
    //
    //   byte b (e.g. 'a' = 0x61 = 0110 0001)
    //     | & 0x7F          -> fold into the ASCII table domain
    //     v
    //   ascii = 0x61
    //     | >> 3            -> index of the bitmap byte (0x61 >> 3 = 12)
    //     v
    //   bitmapByte = bitmap[12]        (bit 1 set iff Pred(0x61))
    //     | & (1 << (ascii & 7))       (0x61 & 7 = 1)
    //     v
    //   bit = bitmapByte & powerOfTwo[1] = bitmapByte & 0x02
    //
    // bit is zero exactly when Pred rejects the byte (or the byte is
    // non-ASCII, which the sign-bit mask below rejects separately).
    const __m256i bitmap = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        reinterpret_cast<const __m128i*>(AsciiBitmap<Pred>::kValues.data())));
    // The second lookup table: index k yields the one-bit mask 1 << k.
    const __m256i powerOfTwo = _mm256_setr_epi8(
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        -128,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        -128,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0);
    const __m256i chunk =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input.data() + i));
    // Fold every byte into 0..127 so the bitmap index stays inside its
    // 16-byte domain. Without this, pshufb would silently truncate the
    // index of a >= 0x80 byte to its low 4 bits and look up an unrelated
    // bitmap entry; non-ASCII bytes are rejected by the sign bits below,
    // not by this lookup.
    const __m256i ascii = _mm256_and_si256(chunk, _mm256_set1_epi8(0x7F));
    // First lookup: byte -> bitmapByte. The shift runs on 16-bit lanes
    // (AVX2 has no byte-wise shift), which lets one byte's low bits leak
    // into its neighbor's high bits, so & 0x1F trims every byte back to
    // a 5-bit index.
    const __m256i bitmapByte = _mm256_shuffle_epi8(
        bitmap,
        _mm256_and_si256(_mm256_srli_epi16(ascii, 3), _mm256_set1_epi8(0x1F)));
    // Second lookup: extract the bit for (ascii & 7) out of bitmapByte.
    const __m256i bit = _mm256_and_si256(
        _mm256_shuffle_epi8(
            powerOfTwo, _mm256_and_si256(ascii, _mm256_set1_epi8(7))),
        bitmapByte);
    // One rejected bit per input byte. The first movemask marks bytes
    // whose lookup bit is zero; the second ORs in the sign bit of every
    // raw byte, so any non-ASCII (>= 0x80) byte is rejected here and
    // re-examined by the classification below, which hands it to UTF-8
    // decoding.
    const uint32_t rejected =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(bit, _mm256_setzero_si256())) |
        _mm256_movemask_epi8(chunk);
    if (rejected != 0) {
      // ctz picks the first rejected byte; classify it for the caller.
      input.remove_prefix(i + __builtin_ctz(rejected));
      const auto b = static_cast<uint8_t>(input.front());
      return b == '%'
          ? ScanStop::kEscape
          : (b >= 0x80 ? ScanStop::kMultibyte : ScanStop::kRejected);
    }
  }
#endif
  for (; i < size; ++i) {
    const auto b = static_cast<uint8_t>(input[i]);
    if (b >= 0x80) {
      input.remove_prefix(i);
      return ScanStop::kMultibyte;
    }
    if (!Pred(b)) {
      input.remove_prefix(i);
      return b == '%' ? ScanStop::kEscape : ScanStop::kRejected;
    }
  }
  input.remove_prefix(size);
  return ScanStop::kEnd;
}

// Why a component scan failed.
enum class ScanResult {
  kValid, //< Every character was accepted.
  kRejected, //< A character was rejected.
  kMalformedEscape, //< A '%' was not followed by two hex digits;
                    //< the URL is invalid as a whole.
};

// Scans component left to right, consuming Pred-accepted characters,
// valid "%XX" escape pairs (when allowEscape), and - outside the
// authority - non-ASCII characters that are neither spaces nor ISO
// controls. rejectedAt, when non-null, receives the tail starting at
// the first unconsumed character. The distinction between kRejected
// and kMalformedEscape only matters to parseAuthority's registry
// fallback; every other caller treats both as an invalid URL.
template <bool (*Pred)(char)>
FOLLY_ALWAYS_INLINE ScanResult scanComponent(
    std::string_view component,
    bool allowEscape,
    std::string_view* rejectedAt = nullptr) {
  auto input = component;
  while (!input.empty()) {
    const auto stop = skipAccepted<Pred>(input);
    if (stop == ScanStop::kEnd) {
      break;
    }
    if (stop == ScanStop::kEscape && allowEscape) {
      if (input.size() < 3 ||
          !isHexDigit(static_cast<unsigned char>(input[1])) ||
          !isHexDigit(static_cast<unsigned char>(input[2]))) {
        if (rejectedAt != nullptr) {
          *rejectedAt = input;
        }
        return ScanResult::kMalformedEscape;
      }
      input.remove_prefix(3);
      continue;
    }
    if (stop == ScanStop::kMultibyte && allowEscape) {
      int length;
      const auto codePoint =
          utf8proc_codepoint(input.data(), input.data() + input.size(), length);
      // Invalid UTF-8 sequence is accepted like any other non-ASCII character
      // and skipped per the replacement granularity: one byte for a bad lead
      // byte or a truncated sequence, the full length for a decoded code point.
      const bool accepted = codePoint < 0 ||
          (!isSpaceChar(codePoint) && !isIsoControl(codePoint));
      if (accepted) {
        input.remove_prefix(codePoint < 0 ? 1 : length);
        continue;
      }
    }
    // Any stop reason not continued above - a rejected character, a '%'
    // without escapes allowed, or a disallowed non-ASCII character -
    // fails the scan.
    if (rejectedAt != nullptr) {
      *rejectedAt = input;
    }
    return ScanResult::kRejected;
  }
  return ScanResult::kValid;
}

// Checks that every character of component is accepted (with escapes
// when allowEscape). Malformed escape pairs fail the check just like
// rejected characters: Spark maps both to a null result.
template <bool (*Pred)(char)>
FOLLY_ALWAYS_INLINE bool checkComponent(
    std::string_view component,
    bool allowEscape) {
  return scanComponent<Pred>(component, allowEscape) == ScanResult::kValid;
}

// Splits the "#fragment" off the back of url, if present. The callers
// have already cut the path and query at '#', so a fragment marker can
// only be the leading character; any '#' later in the fragment is left
// for checkComponent<isUric> to reject.
void splitRef(std::string_view& url, std::optional<std::string_view>& ref) {
  if (!url.empty() && url.front() == '#') {
    ref = url.substr(1);
    url.remove_prefix(url.size());
  }
}

// Splits the "?query" off the front of remainder, if present, and
// validates it with the uric grammar.
bool splitQuery(
    std::string_view& remainder,
    std::optional<std::string_view>& query) {
  if (remainder.empty() || remainder.front() != '?') {
    return true;
  }
  remainder.remove_prefix(1);
  const auto queryEnd = remainder.find('#');
  const auto queryPart = remainder.substr(
      0, queryEnd == std::string_view::npos ? remainder.size() : queryEnd);
  if (!checkComponent<isUric>(queryPart, true)) {
    return false;
  }
  query = queryPart;
  remainder.remove_prefix(queryPart.size());
  return true;
}

// Splits the path off the front of remainder, up to '?' or '#', and
// validates it with the path grammar. An empty path is valid and stored
// as such: parse_url returns '' (not null) for it.
bool splitPath(std::string_view& remainder, detail::ParsedUrl& parsed) {
  const auto pathEnd = remainder.find_first_of("?#");
  const auto path = remainder.substr(
      0, pathEnd == std::string_view::npos ? remainder.size() : pathEnd);
  if (!checkComponent<isPathChar>(path, true)) {
    return false;
  }
  parsed.path = path;
  remainder.remove_prefix(path.size());
  return true;
}

// Checks a decimal port number: digits only (the caller has already
// validated that) and at most INT32_MAX. An empty port ("host:") is
// valid.
bool checkPort(std::string_view port) {
  int64_t value = 0;
  for (const char c : port) {
    value = value * 10 + (c - '0');
    if (value > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  return true;
}

// Scans "d.d.d.d" (each byte 0-255) off the front of input. When
// numericOnly is set the address must span all of input (used inside
// IPv6 literals). Returns false when input does not start with an IPv4
// address; input is left untouched so callers can fall back to hostname
// parsing.
bool scanIpv4Address(
    std::string_view input,
    bool numericOnly,
    std::string_view& out) {
  const auto candidate = leadingRun<isIpv4Char>(input);
  if (candidate.empty() || (numericOnly && candidate.size() != input.size())) {
    return false;
  }
  auto rest = candidate;
  for (int byteIndex = 0; byteIndex < 4; ++byteIndex) {
    const auto digits = leadingRun<isDigit>(rest);
    if (digits.empty()) {
      return false;
    }
    // Leading zeros do not change the value, so a digit run of any length
    // is a valid byte as long as its value fits in 0..255. Accumulate in
    // a wider type and bail out before it wraps: a run like
    // "99999999999999999999" is not an IPv4 byte at all.
    int64_t value = 0;
    for (const char c : digits) {
      value = value * 10 + (c - '0');
      if (value > std::numeric_limits<int32_t>::max()) {
        return false;
      }
    }
    if (value > 255) {
      return false;
    }
    rest.remove_prefix(digits.size());
    if (byteIndex < 3) {
      if (rest.empty() || rest.front() != '.') {
        return false;
      }
      rest.remove_prefix(1);
    }
  }
  if (!rest.empty()) {
    return false;
  }
  out = candidate;
  return true;
}

// Parses a hostname: labels of alphanumerics and dashes separated by
// single dots, stopping at ':' or the end of input. A label may not end
// with a dash, and the last label of a multi-label hostname must start
// with a letter - the rule that leaves "1.2.3" without a host, since its
// last label starts with a digit. Consumes the hostname and stores it in
// parsed.
bool parseHostname(std::string_view& input, detail::ParsedUrl& parsed) {
  const auto host = input;
  std::string_view lastLabel;
  while (true) {
    const auto label = leadingRun<isAlnum>(input);
    if (label.empty()) {
      break;
    }
    lastLabel = label;
    input.remove_prefix(label.size());
    const auto labelTail = leadingRun<isAlnumOrDash>(input);
    if (!labelTail.empty() && labelTail.back() == '-') {
      return false;
    }
    input.remove_prefix(labelTail.size());
    if (input.empty() || input.front() != '.') {
      break;
    }
    input.remove_prefix(1);
    if (input.empty()) {
      break;
    }
  }
  if (!input.empty() && input.front() != ':') {
    return false;
  }
  if (lastLabel.empty()) {
    return false;
  }
  // lastLabel.data() == host.data() means the hostname has exactly one
  // label, so the "last label must start with a letter" rule does not
  // apply.
  if (lastLabel.data() != host.data() && !isAlpha(lastLabel.front())) {
    return false;
  }
  parsed.host = host.substr(0, host.size() - input.size());
  return true;
}

// Scans a run of ':'-separated groups of at most four hex digits.
// startsWithGroup reports whether the input began with a group: it is
// false without failing (consuming nothing) when the input begins with
// "::" or with an embedded IPv4 address, whose handling belongs to the
// callers. After a first group, a missing group is a syntax error. A
// group followed by '.' belongs to an embedded IPv4 address: before
// the first group it makes startsWithGroup false, after it the scan
// stops with the ':' separator unconsumed. byteCount accumulates 2
// per group.
bool scanHexGroups(
    std::string_view& input,
    int& byteCount,
    bool& startsWithGroup) {
  auto group = leadingHexDigits(input);
  const auto ipv4Ahead = !group.empty() && input.size() > group.size() &&
      input[group.size()] == '.';
  if (group.empty() || ipv4Ahead) {
    startsWithGroup = false;
    return true;
  }
  if (group.size() > 4) {
    return false;
  }
  startsWithGroup = true;
  byteCount += 2;
  input.remove_prefix(group.size());
  while (!input.empty() && input.front() == ':' &&
         !(input.size() > 1 && input[1] == ':')) {
    const auto beforeColon = input;
    input.remove_prefix(1);
    group = leadingHexDigits(input);
    if (group.empty()) {
      return false;
    }
    if (input.size() > group.size() && input[group.size()] == '.') {
      // The embedded IPv4 address starts after the ':' we just consumed;
      // give the ':' back so the caller sees it.
      input = beforeColon;
      return true;
    }
    if (group.size() > 4) {
      return false;
    }
    byteCount += 2;
    input.remove_prefix(group.size());
  }
  return true;
}

// Consumes an embedded IPv4 address (numericOnly scan); a failure means
// the surrounding IPv6 literal is malformed.
bool skipIpv4Address(std::string_view& input) {
  std::string_view address;
  if (!scanIpv4Address(input, true, address)) {
    return false;
  }
  input.remove_prefix(address.size());
  return true;
}

// Scans the hex groups (or IPv4 address) after a "::".
bool scanHexTail(std::string_view& input, int& byteCount) {
  if (input.empty()) {
    return true;
  }
  bool startsWithGroup;
  if (!scanHexGroups(input, byteCount, startsWithGroup)) {
    return false;
  }
  if (startsWithGroup) {
    if (!input.empty() && input.front() == ':') {
      input.remove_prefix(1);
      if (!skipIpv4Address(input)) {
        return false;
      }
      byteCount += 4;
    }
    return true;
  }
  if (!skipIpv4Address(input)) {
    return false;
  }
  byteCount += 4;
  return true;
}

// Validates an IPv6 address literal (the bracket content without the
// scope id): up to 16 bytes of hex groups, with at most one "::"
// compression and an optional trailing IPv4 form.
bool parseIpv6Reference(std::string_view literal) {
  int byteCount = 0;
  bool compressed = false;
  bool startsWithGroup;
  auto input = literal;
  if (!scanHexGroups(input, byteCount, startsWithGroup)) {
    return false;
  }
  if (startsWithGroup) {
    if (input.substr(0, 2) == "::") {
      compressed = true;
      input.remove_prefix(2);
      if (!scanHexTail(input, byteCount)) {
        return false;
      }
    } else if (!input.empty() && input.front() == ':') {
      input.remove_prefix(1);
      if (!skipIpv4Address(input)) {
        return false;
      }
      byteCount += 4;
    }
  } else if (input.substr(0, 2) == "::") {
    compressed = true;
    input.remove_prefix(2);
    if (!scanHexTail(input, byteCount)) {
      return false;
    }
  }
  return input.empty() && byteCount <= 16 &&
      ((!compressed && byteCount == 16) || (compressed && byteCount < 16));
}

// Parses the "[ipv6[%scope]]" form of hostAndPort, consuming it up to
// and including the closing bracket; the port check stays in
// parseServer, where the IPv6, IPv4 and hostname branches all rejoin.
// Returns false when the authority is not server-based.
bool parseIpv6Host(std::string_view& hostAndPort, detail::ParsedUrl& parsed) {
  const auto closingBracket = hostAndPort.find(']');
  if (closingBracket == std::string_view::npos) {
    return false;
  }
  const auto literal = hostAndPort.substr(1, closingBracket - 1);
  const auto scopeStart = literal.find('%');
  if (scopeStart != std::string_view::npos) {
    const auto scope = literal.substr(scopeStart + 1);
    if (scope.empty() || !checkComponent<isScopeIdChar>(scope, false)) {
      return false;
    }
    if (!parseIpv6Reference(literal.substr(0, scopeStart))) {
      return false;
    }
  } else if (!parseIpv6Reference(literal)) {
    return false;
  }
  parsed.host = hostAndPort.substr(0, closingBracket + 1);
  hostAndPort.remove_prefix(closingBracket + 1);
  return true;
}

// Parses "[userinfo@]host[:port]" within authority. Returns false when
// the authority is not server-based (the caller falls back to registry
// form).
bool parseServer(std::string_view authority, detail::ParsedUrl& parsed) {
  auto hostAndPort = authority;
  // The userinfo ends at the first '@'.
  const auto userInfoEnd = authority.find('@');
  if (userInfoEnd != std::string_view::npos) {
    const auto userInfo = authority.substr(0, userInfoEnd);
    if (!checkComponent<isUserInfoChar>(userInfo, true)) {
      return false;
    }
    parsed.userInfo = userInfo;
    hostAndPort = authority.substr(userInfoEnd + 1);
  }
  if (!hostAndPort.empty() && hostAndPort.front() == '[') {
    if (!parseIpv6Host(hostAndPort, parsed)) {
      return false;
    }
  } else {
    std::string_view address;
    if (scanIpv4Address(hostAndPort, false, address)) {
      const auto afterAddress = hostAndPort.substr(address.size());
      if (afterAddress.empty() || afterAddress.front() == ':') {
        parsed.host = address;
        hostAndPort.remove_prefix(address.size());
      }
    }
    if (parsed.host.empty()) {
      if (!parseHostname(hostAndPort, parsed)) {
        return false;
      }
    }
  }
  // All three host forms rejoin here for the port check.
  if (hostAndPort.empty()) {
    return true;
  }
  if (hostAndPort.front() != ':') {
    return false;
  }
  hostAndPort.remove_prefix(1);
  const auto portEnd = hostAndPort.find('/');
  const auto port = hostAndPort.substr(
      0, portEnd == std::string_view::npos ? hostAndPort.size() : portEnd);
  if (!checkComponent<isDigit>(port, false) || !checkPort(port)) {
    return false;
  }
  hostAndPort.remove_prefix(port.size());
  return hostAndPort.empty();
}

// Parses an authority, preferring the server-based form: try
// "[userinfo@]host[:port]" first; when that grammar fails, fall back to
// registry form, in which HOST and USERINFO are null but AUTHORITY is
// still returned. A malformed escape pair fails the whole URL before
// the server attempt is even made.
bool parseAuthority(std::string_view authority, detail::ParsedUrl& parsed) {
  // An authority scans with the '%'-permitting server set unless it
  // starts with ']' - counter-intuitively, an authority with no ']' at
  // all still gets the permissive set; only a leading ']' falls back to
  // the strict one.
  // Multi-byte characters that are neither spaces nor ISO controls scan
  // as server characters (and registry characters). In the permissive
  // branch '%' is a set member and never reaches escape validation, so
  // a malformed pair there is only caught by the reg-name scan below.
  const auto serverChars = authority.front() == ']'
      ? scanComponent<isServerChar>(authority, true)
      : scanComponent<isServerPercentChar>(authority, true);
  if (serverChars == ScanResult::kMalformedEscape) {
    return false;
  }
  const auto regChars = scanComponent<isRegNameChar>(authority, true);
  if (regChars == ScanResult::kMalformedEscape) {
    return false;
  }
  if (serverChars == ScanResult::kValid && parseServer(authority, parsed)) {
    parsed.authority = authority;
    return true;
  }
  // Registry-based fallback: drop whatever a failed server attempt
  // stored, but keep the raw authority.
  parsed.host = std::string_view{};
  parsed.userInfo = std::nullopt;
  if (regChars == ScanResult::kValid) {
    parsed.authority = authority;
    return true;
  }
  return false;
}

// Parses the "//authority" prefix, the path, and the query of a
// hierarchical URL, consuming remainder in place.
bool parseHierarchical(std::string_view& remainder, detail::ParsedUrl& parsed) {
  if (remainder.substr(0, 2) == "//") {
    remainder.remove_prefix(2);
    // The authority ends at the first '/', '?' or '#'.
    const auto authorityEnd = remainder.find_first_of("/?#");
    const auto authority = remainder.substr(
        0,
        authorityEnd == std::string_view::npos ? remainder.size()
                                               : authorityEnd);
    if (authority.empty()) {
      if (authorityEnd == std::string_view::npos) {
        // '//' with nothing after it fails to parse entirely.
        return false;
      }
      // An empty authority before a path, query, or fragment is valid.
    } else if (!parseAuthority(authority, parsed)) {
      return false;
    }
    remainder.remove_prefix(authority.size());
  }
  return splitPath(remainder, parsed) && splitQuery(remainder, parsed.query);
}

} // namespace

namespace detail {

bool parseUrl(std::string_view url, ParsedUrl& parsed) {
  auto remainder = url;

  // The scheme is everything before the first ':' that appears before
  // any '/', '?' or '#'.
  const auto schemeEnd = remainder.find_first_of(":/?#");
  if (schemeEnd != std::string_view::npos && remainder[schemeEnd] == ':') {
    if (schemeEnd == 0) {
      return false;
    }
    const auto scheme = remainder.substr(0, schemeEnd);
    if (!isAlpha(scheme.front()) ||
        !checkComponent<isSchemeChar>(scheme.substr(1), false)) {
      return false;
    }
    parsed.protocol = scheme;
    remainder.remove_prefix(schemeEnd + 1);
    if (remainder.empty() || remainder.front() != '/') {
      // Opaque URI: everything before '#' is the scheme-specific part;
      // there is no path, query, or authority at all.
      const auto opaqueEnd = remainder.find('#');
      const auto opaquePart = remainder.substr(
          0,
          opaqueEnd == std::string_view::npos ? remainder.size() : opaqueEnd);
      if (opaquePart.empty() || !checkComponent<isUric>(opaquePart, true)) {
        return false;
      }
      remainder.remove_prefix(opaquePart.size());
      splitRef(remainder, parsed.ref);
      if (parsed.ref.has_value() &&
          !checkComponent<isUric>(*parsed.ref, true)) {
        return false;
      }
      return remainder.empty();
    }
  }

  if (!parseHierarchical(remainder, parsed)) {
    return false;
  }
  splitRef(remainder, parsed.ref);
  if (parsed.ref.has_value() && !checkComponent<isUric>(*parsed.ref, true)) {
    return false;
  }
  return remainder.empty();
}

} // namespace detail
} // namespace facebook::velox::functions::sparksql
