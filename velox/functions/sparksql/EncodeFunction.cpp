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

#include "velox/functions/sparksql/EncodeFunction.h"

#include <algorithm>
#include <array>
#include <compare>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <unicode/ucnv.h>
#include <unicode/ucnv_cb.h>
#include <unicode/ucnv_err.h>
#include <unicode/uenum.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/lib/Utf8Utils.h"

namespace facebook::velox::functions::sparksql::detail {
namespace {

constexpr char32_t kReplacementCodePoint = 0xFFFD;

std::strong_ordering compareIgnoreCase(
    const StringView& value,
    std::string_view expected) {
  const auto compareCharacter = [](char left, char right) {
    const auto toUpper = [](char character) {
      return character >= 'a' && character <= 'z'
          ? static_cast<char>(character - ('a' - 'A'))
          : character;
    };
    return toUpper(left) <=> toUpper(right);
  };
  const auto commonSize = std::min(value.size(), expected.size());
  for (size_t i = 0; i < commonSize; ++i) {
    if (const auto comparison = compareCharacter(value.data()[i], expected[i]);
        comparison != 0) {
      return comparison;
    }
  }
  return value.size() <=> expected.size();
}

bool equalsIgnoreCase(const StringView& value, std::string_view expected) {
  return compareIgnoreCase(value, expected) == 0;
}

size_t decodeCodePoint(
    const char* input,
    size_t inputSize,
    size_t position,
    char32_t& codePoint) {
  int32_t decodedCodePoint;
  const auto bytesConsumed = decodeUtf8CodePointOrReplacement(
      input + position, inputSize - position, decodedCodePoint);
  codePoint = static_cast<char32_t>(decodedCodePoint);
  return bytesConsumed;
}

size_t utf8Length(char32_t codePoint) {
  if (codePoint <= 0x7F) {
    return 1;
  }
  if (codePoint <= 0x7FF) {
    return 2;
  }
  if (codePoint <= 0xFFFF) {
    return 3;
  }
  return 4;
}

size_t writeUtf8(char32_t codePoint, char* output) {
  if (codePoint <= 0x7F) {
    output[0] = static_cast<char>(codePoint);
    return 1;
  }
  if (codePoint <= 0x7FF) {
    output[0] = static_cast<char>(0xC0 | (codePoint >> 6));
    output[1] = static_cast<char>(0x80 | (codePoint & 0x3F));
    return 2;
  }
  if (codePoint <= 0xFFFF) {
    output[0] = static_cast<char>(0xE0 | (codePoint >> 12));
    output[1] = static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
    output[2] = static_cast<char>(0x80 | (codePoint & 0x3F));
    return 3;
  }
  output[0] = static_cast<char>(0xF0 | (codePoint >> 18));
  output[1] = static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F));
  output[2] = static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
  output[3] = static_cast<char>(0x80 | (codePoint & 0x3F));
  return 4;
}

size_t writeUtf16(char32_t codePoint, char* output, bool bigEndian) {
  auto writeUnit = [&](char16_t unit, char* destination) {
    if (bigEndian) {
      destination[0] = static_cast<char>((unit >> 8) & 0xFF);
      destination[1] = static_cast<char>(unit & 0xFF);
    } else {
      destination[0] = static_cast<char>(unit & 0xFF);
      destination[1] = static_cast<char>((unit >> 8) & 0xFF);
    }
  };

  if (codePoint <= 0xFFFF) {
    writeUnit(static_cast<char16_t>(codePoint), output);
    return 2;
  }
  const auto adjusted = codePoint - 0x10000;
  writeUnit(static_cast<char16_t>(0xD800 + (adjusted >> 10)), output);
  writeUnit(static_cast<char16_t>(0xDC00 + (adjusted & 0x3FF)), output + 2);
  return 4;
}

void writeUtf32(char32_t codePoint, char* output, bool bigEndian) {
  for (size_t i = 0; i < 4; ++i) {
    const auto shift = bigEndian ? 24 - 8 * i : 8 * i;
    output[i] = static_cast<char>((codePoint >> shift) & 0xFF);
  }
}

Status unmappableCharacter(const StringView& charset) {
  return Status::UserError(
      "encode: input contains a character that cannot be encoded using '{}'",
      std::string(charset.data(), charset.size()));
}

// Returns the canonical Java name for aliases omitted by ICU.
std::string_view canonicalSpecialJavaCharset(const StringView& requestedName) {
  static constexpr std::string_view kIbm300Names[] = {
      "300", "CP300", "IBM-300", "IBM300", "X-IBM300"};
  for (const auto& name : kIbm300Names) {
    if (equalsIgnoreCase(requestedName, name)) {
      return "X-IBM300";
    }
  }

  static constexpr std::string_view kIbm834Names[] = {
      "834", "CP834", "IBM-834", "IBM834", "X-IBM834"};
  for (const auto& name : kIbm834Names) {
    if (equalsIgnoreCase(requestedName, name)) {
      return "X-IBM834";
    }
  }
  return {};
}

// ICU's JAVA alias standard identifies names exposed by Charset.forName. A
// few JDK charsets have no JAVA alias in ICU, so retain their canonical Java
// names as explicit compatibility exceptions.
bool isJavaCharset(const StringView& requestedName, const char* canonicalName) {
  static constexpr std::string_view kIcuJavaAliasesMissingFromJdk[] = {
      "IBM1141_LF", "IBM1153",         "EBCDIC-IS",   "X-IBM1153",
      "X-IBM1363",  "X-IBM1363C",      "X-IBM1371",   "X-IBM1388",
      "X-IBM1390",  "X-IBM1399",       "X-IBM33722A", "X-IBM33722C",
      "X-IBM720",   "X-IBM867",        "X-IBM930A",   "X-IBM939A",
      "X-IBM954",   "X-IBM954C",       "X-IBM971",    "X-ISO-8859-6S",
      "X-KSC5601",  "X-COMPOUND-TEXT", "X-ROMAN8",    "X-WINDOWS-1256S",
  };
  for (const auto& unsupported : kIcuJavaAliasesMissingFromJdk) {
    if (equalsIgnoreCase(requestedName, unsupported)) {
      return false;
    }
  }

  if (!canonicalSpecialJavaCharset(requestedName).empty()) {
    return true;
  }

  UErrorCode error{U_ZERO_ERROR};
  std::unique_ptr<UEnumeration, decltype(&uenum_close)> javaNames{
      ucnv_openStandardNames(canonicalName, "JAVA", &error), &uenum_close};
  int32_t nameLength;
  while (U_SUCCESS(error)) {
    const char* name = uenum_next(javaNames.get(), &nameLength, &error);
    if (name == nullptr) {
      break;
    }
    if (equalsIgnoreCase(
            requestedName,
            std::string_view{name, static_cast<size_t>(nameLength)})) {
      return true;
    }
  }

  // These OpenJDK aliases are absent from ICU 72's JAVA-standard enumeration
  // even though ICU can open their converters.
  static constexpr std::string_view kJavaAliasesMissingFromIcu = R"(500
855
921
ANSI-1251
ANSI_X3.4-1968
ANSI_X3.4-1986
BIG5_HKSCS
BIG5HKSCS
BIG5-HKSCS:UNICODE3.0
CESU-8
CESU8
CNS11643
CP290
CP367
CP921
CP932
CP-AR
CSASCII
CSBIG5
CSEUCKR
CSIBM278
CSIBM284
CSIBM285
CSIBM290
CSIBM297
CSIBM420
CSIBM424
CSIBM500
CSIBM868
CSIBM870
CSISO2022JP2
CSISOLATIN3
CSJISENCODING
EBCDIC-CP-AR1
EBCDIC-CP-AR2
EBCDIC-CP-CH
EBCDIC-CP-FR
EBCDIC-CP-GB
EBCDIC-CP-HE
EBCDIC-CP-ROECE
EBCDIC-CP-SE
EBCDIC-CP-YU
EBCDIC-DE-273+EURO
EBCDIC-DK-277+EURO
EBCDIC-ES-284+EURO
EBCDIC-FI-278+EURO
EBCDIC-GB-285+EURO
EBCDIC-INTERNATIONAL-500+EURO
EBCDIC-IT-280+EURO
EBCDIC-JP-KANA
EBCDIC-NO-277+EURO
EBCDIC-SE-278+EURO
EBCDIC-US-037+EURO
EUC-CN
EUC_CN
EUCCN
EUC_JP
EUCJP
EUC_KR
EUCKR
EUC-TW
EUC_TW
EUCTW
GB18030
GB2312
GB2312-1980
GB2312-80
IBM-037
IBM-1006
IBM1025
IBM-1026
IBM-1047
IBM1089
IBM1097
IBM-1098
IBM1112
IBM1122
IBM1123
IBM1124
IBM-1129
IBM1129
IBM-1140
IBM1140
IBM-1141
IBM1141
IBM-1142
IBM1142
IBM-1143
IBM1143
IBM-1144
IBM1144
IBM-1145
IBM1145
IBM-1146
IBM1146
IBM-1147
IBM1147
IBM-1148
IBM1148
IBM-1149
IBM1149
IBM-1252
IBM1252
IBM-1364
IBM1364
IBM1383
IBM-273
IBM-277
IBM-278
IBM-280
IBM-284
IBM-285
IBM-290
IBM290
IBM-297
IBM33722
IBM-33722_VASCII_VPUA
IBM367
IBM-37
IBM-420
IBM-424
IBM-437
IBM-500
IBM-5050
IBM-737
IBM-775
IBM813
IBM819
IBM-838
IBM-850
IBM-852
IBM-855
IBM-856
IBM-857
IBM-858
IBM858
IBM-860
IBM-861
IBM-862
IBM-863
IBM-864
IBM-865
IBM-866
IBM-868
IBM-869
IBM-870
IBM-871
IBM874
IBM-875
IBM912
IBM913
IBM914
IBM915
IBM916
IBM-918
IBM920
IBM-921
IBM921
IBM-922
IBM923
IBM-930
IBM-932
IBM932
IBM933
IBM935
IBM937
IBM-939
IBM-942
IBM942
IBM943
IBM-943C
IBM943C
IBM949
IBM949C
IBM950
IBM964
IBM970
IBM-EUCCN
IBMEUCCN
IBM-EUCJP
IBM-EUCTW
ISCII
ISO-10646-UCS-2
ISO2022CN
ISO-2022-CN-CNS
ISO2022CN_CNS
ISO2022JP
ISO-2022-JP-2
ISO2022JP2
ISO2022KR
ISO_646.IRV:1991
ISO_8859-1
ISO_8859_1
ISO8859-1
ISO8859_1
ISO-8859-11
ISO8859_11
ISO_8859-13
ISO8859-13
ISO8859_13
ISO_8859-15
ISO8859-15
ISO8859_15
ISO_8859-2
ISO8859-2
ISO8859_2
ISO_8859-3
ISO8859-3
ISO8859_3
ISO_8859-4
ISO8859-4
ISO8859_4
ISO_8859-5
ISO8859-5
ISO8859_5
ISO_8859-6
ISO8859-6
ISO8859_6
ISO_8859-7
ISO8859-7
ISO8859_7
ISO_8859-8
ISO8859-8
ISO8859_8
ISO_8859-9
ISO8859-9
ISO8859_9
ISO_8859-9:1989
ISO-IR-6
JIS
JIS_ENCODING
KOI8_R
KOI8-U
KOI8_U
KSC_5601
KSC5601
KS_C_5601-1987
KSC5601-1987
KSC5601_1987
L9
LATIN-9
LATIN9
MACCENTRALEUROPE
MACCYRILLIC
MS-874
MS932
MS_936
MS936
MS_949
MS950
PCK
PC-MULTILINGUAL-850+EURO
SHIFT-JIS
SJIS
SUN_EU_GREEK
TIS-620
TIS620
UNICODE
UNICODE-1-1-UTF-8
US
UTF_16
UTF16
UTF_16BE
UTF_16LE
UTF-32
UTF_32
UTF32
UTF-32BE
UTF_32BE
UTF-32LE
UTF_32LE
UTF8
WINDOWS-437
WINDOWS-932
WINDOWS949
WINDOWS-950
X-EUC-CN
X-EUCJP
X-EUC-TW
X-IBM1129
X-IBM1383
X-IBM932
X-IBM943C
X-PCK
X-WINDOWS-949)";
  static const std::unordered_set<std::string_view> kJavaAliasIndex = [] {
    std::unordered_set<std::string_view> names;
    size_t nameStart{0};
    while (nameStart < kJavaAliasesMissingFromIcu.size()) {
      auto nameEnd = kJavaAliasesMissingFromIcu.find('\n', nameStart);
      if (nameEnd == std::string_view::npos) {
        nameEnd = kJavaAliasesMissingFromIcu.size();
      }
      names.emplace(
          kJavaAliasesMissingFromIcu.substr(nameStart, nameEnd - nameStart));
      nameStart = nameEnd + 1;
    }
    return names;
  }();
  std::string normalizedName{requestedName.data(), requestedName.size()};
  std::transform(
      normalizedName.begin(),
      normalizedName.end(),
      normalizedName.begin(),
      [](char character) {
        return character >= 'a' && character <= 'z'
            ? static_cast<char>(character - ('a' - 'A'))
            : character;
      });
  return kJavaAliasIndex.contains(std::string_view{normalizedName});
}

bool isJapaneseIso2022(const char* canonicalName) {
  const StringView name{canonicalName};
  return equalsIgnoreCase(name, "ISO_2022,LOCALE=JA,VERSION=0") ||
      equalsIgnoreCase(name, "ISO_2022,LOCALE=JA,VERSION=1") ||
      equalsIgnoreCase(name, "ISO_2022,LOCALE=JA,VERSION=2");
}

bool isJavaShiftJis(const StringView& charset) {
  static constexpr std::string_view kNames[] = {
      "CSSHIFTJIS",
      "MS_KANJI",
      "SHIFT-JIS",
      "SHIFT_JIS",
      "SJIS",
      "X-SJIS",
  };
  for (const auto& name : kNames) {
    if (equalsIgnoreCase(charset, name)) {
      return true;
    }
  }
  return false;
}

bool isJis0208(const StringView& charset) {
  static constexpr std::string_view kNames[] = {
      "X-JIS0208",
      "JIS0208",
      "JIS_C6226-1983",
      "JIS_X0208-1983",
      "CSISO87JISX0208",
      "ISO-IR-87",
      "X0208",
  };
  for (const auto& name : kNames) {
    if (equalsIgnoreCase(charset, name)) {
      return true;
    }
  }
  return false;
}

bool isJis0212(const StringView& charset) {
  static constexpr std::string_view kNames[] = {
      "JIS_X0212-1990",
      "JIS0212",
      "CSISO159JISX02121990",
      "ISO-IR-159",
      "X0212",
  };
  for (const auto& name : kNames) {
    if (equalsIgnoreCase(charset, name)) {
      return true;
    }
  }
  return false;
}

std::string_view javaSubstitutionBytes(
    const StringView& requestedName,
    const char* canonicalName) {
  const auto specialJavaCharset = canonicalSpecialJavaCharset(requestedName);
  if (specialJavaCharset == "X-IBM300") {
    return "\x42\x6F";
  }
  if (specialJavaCharset == "X-IBM834") {
    return "\xFE\xFE";
  }

  UErrorCode error{U_ZERO_ERROR};
  const char* javaName = ucnv_getStandardName(canonicalName, "JAVA", &error);
  const StringView name{javaName == nullptr ? canonicalName : javaName};
  if (equalsIgnoreCase(name, "X-IBM1364") || equalsIgnoreCase(name, "CP930") ||
      equalsIgnoreCase(name, "CP933") || equalsIgnoreCase(name, "CP935") ||
      equalsIgnoreCase(name, "CP937") || equalsIgnoreCase(name, "CP939")) {
    return "\x6F";
  }
  return "?";
}

void writeJavaSubstitutionBytes(
    const void* context,
    UConverterFromUnicodeArgs* args,
    const UChar* /*codeUnits*/,
    int32_t /*length*/,
    UChar32 /*codePoint*/,
    UConverterCallbackReason reason,
    UErrorCode* error) {
  if (reason > UCNV_IRREGULAR) {
    return;
  }
  const auto substitution = *static_cast<const std::string_view*>(context);
  *error = U_ZERO_ERROR;
  ucnv_cbFromUWriteBytes(
      args,
      substitution.data(),
      static_cast<int32_t>(substitution.size()),
      0,
      error);
}

std::vector<UChar> toUtf16(const StringView& input) {
  std::vector<UChar> utf16;
  utf16.reserve(input.size());
  size_t inputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    if (codePoint <= 0xFFFF) {
      utf16.push_back(static_cast<UChar>(codePoint));
    } else {
      const auto adjusted = codePoint - 0x10000;
      utf16.push_back(static_cast<UChar>(0xD800 + (adjusted >> 10)));
      utf16.push_back(static_cast<UChar>(0xDC00 + (adjusted & 0x3FF)));
    }
  }
  return utf16;
}

enum class JisDoubleBytePlane {
  k0208,
  k0212,
};

struct JisMapping {
  char32_t codePoint;
  uint16_t bytes;
};

// ICU's versioned ISO-2022-JP mappings are nearly identical to OpenJDK's
// generated JIS_X_0208 and JIS_X_0212 charsets. These corrections preserve
// OpenJDK's exact mapping membership.
constexpr std::array<JisMapping, 7> kJis0208MappingsMissingFromIcu{{
    {0x00A2, 0x2171},
    {0x00A3, 0x2172},
    {0x00AC, 0x224C},
    {0x2014, 0x213D},
    {0x2016, 0x2142},
    {0x2212, 0x215D},
    {0x301C, 0x2141},
}};

constexpr std::array<char32_t, 82> kJis0208MappingsMissingFromOpenJdk{{
    0x2015, 0x2116, 0x2121, 0x2160, 0x2161, 0x2162, 0x2163, 0x2164, 0x2165,
    0x2166, 0x2167, 0x2168, 0x2169, 0x2211, 0x221F, 0x2225, 0x222E, 0x22BF,
    0x2460, 0x2461, 0x2462, 0x2463, 0x2464, 0x2465, 0x2466, 0x2467, 0x2468,
    0x2469, 0x246A, 0x246B, 0x246C, 0x246D, 0x246E, 0x246F, 0x2470, 0x2471,
    0x2472, 0x2473, 0x301D, 0x301F, 0x3231, 0x3232, 0x3239, 0x32A4, 0x32A5,
    0x32A6, 0x32A7, 0x32A8, 0x3303, 0x330D, 0x3314, 0x3318, 0x3322, 0x3323,
    0x3326, 0x3327, 0x332B, 0x3336, 0x333B, 0x3349, 0x334A, 0x334D, 0x3351,
    0x3357, 0x337B, 0x337C, 0x337D, 0x337E, 0x338E, 0x338F, 0x339C, 0x339D,
    0x339E, 0x33A1, 0x33C4, 0x33CD, 0xF86F, 0xFF0D, 0xFF5E, 0xFFE0, 0xFFE1,
    0xFFE2,
}};

constexpr std::array<JisMapping, 2> kJis0212MappingsMissingFromIcu{{
    {0x2116, 0x2271},
    {0xFF5E, 0x2237},
}};

template <size_t N>
uint16_t findJisMapping(
    const std::array<JisMapping, N>& mappings,
    char32_t codePoint) {
  const auto it = std::lower_bound(
      mappings.begin(),
      mappings.end(),
      codePoint,
      [](const JisMapping& mapping, char32_t value) {
        return mapping.codePoint < value;
      });
  return it != mappings.end() && it->codePoint == codePoint ? it->bytes : 0;
}

void appendJisBytes(std::string& output, uint16_t bytes) {
  output.push_back(static_cast<char>(bytes >> 8));
  output.push_back(static_cast<char>(bytes & 0xFF));
}

Status encodeJisDoubleByte(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    JisDoubleBytePlane plane,
    bool replaceUnmappable) {
  if (input.empty()) {
    result.resize(0);
    return Status::OK();
  }

  UErrorCode error{U_ZERO_ERROR};
  const char* converterName = plane == JisDoubleBytePlane::k0208
      ? "ISO_2022,locale=ja,version=0"
      : "ISO_2022,locale=ja,version=1";
  std::unique_ptr<UConverter, decltype(&ucnv_close)> converter{
      ucnv_open(converterName, &error), &ucnv_close};
  if (U_FAILURE(error)) {
    return Status::UserError(
        "encode: unsupported charset '{}'",
        std::string(charset.data(), charset.size()));
  }
  ucnv_setFromUCallBack(
      converter.get(),
      UCNV_FROM_U_CALLBACK_STOP,
      nullptr,
      nullptr,
      nullptr,
      &error);
  if (U_FAILURE(error)) {
    return Status::UserError(
        "encode: unsupported charset '{}'",
        std::string(charset.data(), charset.size()));
  }

  std::string output;
  output.reserve(input.size());
  size_t inputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);

    const auto mapping = plane == JisDoubleBytePlane::k0208
        ? findJisMapping(kJis0208MappingsMissingFromIcu, codePoint)
        : findJisMapping(kJis0212MappingsMissingFromIcu, codePoint);
    if (mapping != 0) {
      appendJisBytes(output, mapping);
      continue;
    }

    const bool rejectedJis0208Mapping = plane == JisDoubleBytePlane::k0208 &&
        std::binary_search(kJis0208MappingsMissingFromOpenJdk.begin(),
                           kJis0208MappingsMissingFromOpenJdk.end(),
                           codePoint);

    UChar utf16[2];
    int32_t numUtf16Units;
    if (codePoint <= 0xFFFF) {
      utf16[0] = static_cast<UChar>(codePoint);
      numUtf16Units = 1;
    } else {
      const auto adjusted = codePoint - 0x10000;
      utf16[0] = static_cast<UChar>(0xD800 + (adjusted >> 10));
      utf16[1] = static_cast<UChar>(0xDC00 + (adjusted & 0x3FF));
      numUtf16Units = 2;
    }

    char iso2022Bytes[16]{};
    error = U_ZERO_ERROR;
    ucnv_resetFromUnicode(converter.get());
    const auto numIso2022Bytes = ucnv_fromUChars(
        converter.get(),
        iso2022Bytes,
        sizeof(iso2022Bytes),
        utf16,
        numUtf16Units,
        &error);
    const bool isJis0208Sequence = plane == JisDoubleBytePlane::k0208 &&
        numIso2022Bytes == 8 &&
        std::memcmp(iso2022Bytes, "\x1B\x24\x42", 3) == 0 &&
        std::memcmp(iso2022Bytes + 5, "\x1B\x28\x42", 3) == 0;
    const bool isJis0212Sequence = plane == JisDoubleBytePlane::k0212 &&
        numIso2022Bytes == 9 &&
        std::memcmp(iso2022Bytes, "\x1B\x24\x28\x44", 4) == 0 &&
        std::memcmp(iso2022Bytes + 6, "\x1B\x28\x42", 3) == 0;
    const bool isExpectedPlane = !rejectedJis0208Mapping && U_SUCCESS(error) &&
        (isJis0208Sequence || isJis0212Sequence);
    if (!isExpectedPlane) {
      if (!replaceUnmappable) {
        return unmappableCharacter(charset);
      }
      if (plane == JisDoubleBytePlane::k0208) {
        output.append("\x21\x29", 2);
      } else {
        output.append("\x22\x44", 2);
      }
      continue;
    }

    const auto dataOffset = plane == JisDoubleBytePlane::k0208 ? 3 : 4;
    output.append(iso2022Bytes + dataOffset, 2);
  }

  result.resize(output.size());
  std::memcpy(result.data(), output.data(), output.size());
  return Status::OK();
}

Status encodeLegacy(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    bool replaceUnmappable) {
  if (input.empty()) {
    result.resize(0);
    return Status::OK();
  }

  const std::string charsetName{charset.data(), charset.size()};
  const char* converterName =
      isJavaShiftJis(charset) ? "ibm-943_P130-1999" : charsetName.c_str();
  std::string_view callbackSubstitution;
  UErrorCode error{U_ZERO_ERROR};
  std::unique_ptr<UConverter, decltype(&ucnv_close)> converter{
      ucnv_open(converterName, &error), &ucnv_close};
  // resolveCharset() already validated that the charset opens, but a
  // user-provided name must never trigger a fatal check here, so degrade
  // gracefully to a catchable user error.
  if (U_FAILURE(error)) {
    return Status::UserError("encode: unsupported charset '{}'", charsetName);
  }

  if (replaceUnmappable) {
    UErrorCode nameError{U_ZERO_ERROR};
    const char* canonicalName = ucnv_getName(converter.get(), &nameError);
    if (U_FAILURE(nameError) || canonicalName == nullptr) {
      return Status::UserError("encode: unsupported charset '{}'", charsetName);
    }
    callbackSubstitution = javaSubstitutionBytes(charset, canonicalName);
    error = U_ZERO_ERROR;
    const bool writesRawSubstitution = callbackSubstitution != "?";
    ucnv_setFromUCallBack(
        converter.get(),
        writesRawSubstitution ? writeJavaSubstitutionBytes
                              : UCNV_FROM_U_CALLBACK_SUBSTITUTE,
        writesRawSubstitution ? &callbackSubstitution : nullptr,
        nullptr,
        nullptr,
        &error);
    if (U_FAILURE(error)) {
      return Status::UserError("encode: unsupported charset '{}'", charsetName);
    }
    UErrorCode substError{U_ZERO_ERROR};
    if (isJapaneseIso2022(canonicalName)) {
      // Java uses JIS bytes 21 29. Set the corresponding Unicode character so
      // ICU emits the required ISO-2022 shift sequences around the replacement.
      const UChar substitution = 0xFF1F;
      ucnv_setSubstString(converter.get(), &substitution, 1, &substError);
    } else if (
        !writesRawSubstitution && ucnv_getMinCharSize(converter.get()) == 1) {
      ucnv_setSubstChars(
          converter.get(),
          callbackSubstitution.data(),
          static_cast<int8_t>(callbackSubstitution.size()),
          &substError);
    }
    if (U_FAILURE(substError)) {
      return Status::UserError("encode: unsupported charset '{}'", charsetName);
    }
  } else {
    error = U_ZERO_ERROR;
    ucnv_setFromUCallBack(
        converter.get(),
        UCNV_FROM_U_CALLBACK_STOP,
        nullptr,
        nullptr,
        nullptr,
        &error);
    if (U_FAILURE(error)) {
      return Status::UserError("encode: unsupported charset '{}'", charsetName);
    }
  }

  const auto utf16 = toUtf16(input);
  const auto* utf16Data = utf16.empty() ? nullptr : utf16.data();
  error = U_ZERO_ERROR;
  const auto outputSize = ucnv_fromUChars(
      converter.get(),
      nullptr,
      0,
      utf16Data,
      static_cast<int32_t>(utf16.size()),
      &error);
  if (error != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(error)) {
    return unmappableCharacter(charset);
  }

  result.resize(outputSize);
  if (outputSize == 0) {
    return Status::OK();
  }
  error = U_ZERO_ERROR;
  ucnv_resetFromUnicode(converter.get());
  const auto bytesWritten = ucnv_fromUChars(
      converter.get(),
      result.data(),
      outputSize,
      utf16Data,
      static_cast<int32_t>(utf16.size()),
      &error);
  if (U_FAILURE(error)) {
    return unmappableCharacter(charset);
  }
  VELOX_CHECK_EQ(bytesWritten, outputSize);
  return Status::OK();
}

Status encodeUtf8(exec::StringWriter& result, const StringView& input) {
  size_t outputSize{0};
  size_t position{0};
  bool isValid{true};
  while (position < input.size()) {
    char32_t codePoint;
    const auto bytesConsumed =
        decodeCodePoint(input.data(), input.size(), position, codePoint);
    if (codePoint == kReplacementCodePoint &&
        !(bytesConsumed == 3 &&
          static_cast<unsigned char>(input.data()[position]) == 0xEF &&
          static_cast<unsigned char>(input.data()[position + 1]) == 0xBF &&
          static_cast<unsigned char>(input.data()[position + 2]) == 0xBD)) {
      isValid = false;
    }
    outputSize += utf8Length(codePoint);
    position += bytesConsumed;
  }

  result.resize(outputSize);
  if (isValid) {
    if (!input.empty()) {
      std::memcpy(result.data(), input.data(), input.size());
    }
    return Status::OK();
  }

  position = 0;
  size_t outputPosition{0};
  while (position < input.size()) {
    char32_t codePoint;
    const auto bytesConsumed =
        decodeCodePoint(input.data(), input.size(), position, codePoint);
    outputPosition += writeUtf8(codePoint, result.data() + outputPosition);
    position += bytesConsumed;
  }
  return Status::OK();
}

Status encodeSingleByte(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    char32_t limit,
    bool replaceUnmappable) {
  result.resize(input.size());
  size_t inputPosition{0};
  size_t outputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    if (codePoint >= limit) {
      if (!replaceUnmappable) {
        return unmappableCharacter(charset);
      }
      codePoint = '?';
    }
    result.data()[outputPosition++] = static_cast<char>(codePoint);
  }
  result.resize(outputPosition);
  return Status::OK();
}

Status encodeUtf16(
    exec::StringWriter& result,
    const StringView& input,
    bool bigEndian,
    bool includeBom) {
  if (input.empty()) {
    result.resize(0);
    return Status::OK();
  }
  size_t outputSize{includeBom ? 2u : 0u};
  size_t position{0};
  while (position < input.size()) {
    char32_t codePoint;
    position +=
        decodeCodePoint(input.data(), input.size(), position, codePoint);
    outputSize += codePoint <= 0xFFFF ? 2 : 4;
  }
  result.resize(outputSize);
  size_t outputPosition{0};
  if (includeBom) {
    result.data()[outputPosition++] =
        static_cast<char>(bigEndian ? 0xFE : 0xFF);
    result.data()[outputPosition++] =
        static_cast<char>(bigEndian ? 0xFF : 0xFE);
  }
  size_t inputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    outputPosition +=
        writeUtf16(codePoint, result.data() + outputPosition, bigEndian);
  }
  return Status::OK();
}

Status encodeUtf32(
    exec::StringWriter& result,
    const StringView& input,
    bool bigEndian) {
  size_t numCodePoints{0};
  size_t inputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    ++numCodePoints;
  }
  result.resize(numCodePoints * 4);
  inputPosition = 0;
  size_t outputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    writeUtf32(codePoint, result.data() + outputPosition, bigEndian);
    outputPosition += 4;
  }
  return Status::OK();
}

} // namespace

CharsetType resolveCharset(const StringView& charset, bool legacyJavaCharsets) {
  if (equalsIgnoreCase(charset, "UTF-8")) {
    return CharsetType::kUtf8;
  }
  if (equalsIgnoreCase(charset, "US-ASCII")) {
    return CharsetType::kUsAscii;
  }
  if (equalsIgnoreCase(charset, "ISO-8859-1")) {
    return CharsetType::kIso8859_1;
  }
  if (equalsIgnoreCase(charset, "UTF-16")) {
    return CharsetType::kUtf16;
  }
  if (equalsIgnoreCase(charset, "UTF-16BE")) {
    return CharsetType::kUtf16BE;
  }
  if (equalsIgnoreCase(charset, "UTF-16LE")) {
    return CharsetType::kUtf16LE;
  }
  if (equalsIgnoreCase(charset, "UTF-32")) {
    return CharsetType::kUtf32;
  }
  if (!legacyJavaCharsets) {
    return CharsetType::kUnsupported;
  }

  if (equalsIgnoreCase(charset, "UTF8")) {
    return CharsetType::kUtf8;
  }
  if (equalsIgnoreCase(charset, "ASCII") ||
      equalsIgnoreCase(charset, "US_ASCII")) {
    return CharsetType::kUsAscii;
  }
  if (equalsIgnoreCase(charset, "LATIN1") ||
      equalsIgnoreCase(charset, "ISO8859_1") ||
      equalsIgnoreCase(charset, "ISO_8859_1") ||
      equalsIgnoreCase(charset, "ISO8859-1")) {
    return CharsetType::kIso8859_1;
  }
  if (equalsIgnoreCase(charset, "UTF16") ||
      equalsIgnoreCase(charset, "UTF_16") ||
      equalsIgnoreCase(charset, "UNICODE") ||
      equalsIgnoreCase(charset, "UNICODEBIG")) {
    return CharsetType::kUtf16;
  }
  if (equalsIgnoreCase(charset, "UTF16BE") ||
      equalsIgnoreCase(charset, "UTF_16BE") ||
      equalsIgnoreCase(charset, "X-UTF-16BE") ||
      equalsIgnoreCase(charset, "ISO-10646-UCS-2") ||
      equalsIgnoreCase(charset, "UNICODEBIGUNMARKED")) {
    return CharsetType::kUtf16BE;
  }
  if (equalsIgnoreCase(charset, "UTF16LE") ||
      equalsIgnoreCase(charset, "UTF_16LE") ||
      equalsIgnoreCase(charset, "X-UTF-16LE") ||
      equalsIgnoreCase(charset, "UNICODELITTLEUNMARKED")) {
    return CharsetType::kUtf16LE;
  }
  if (equalsIgnoreCase(charset, "UNICODELITTLE")) {
    return CharsetType::kUtf16LEWithBom;
  }
  if (equalsIgnoreCase(charset, "UTF-32BE") ||
      equalsIgnoreCase(charset, "UTF32BE") ||
      equalsIgnoreCase(charset, "UTF_32BE") ||
      equalsIgnoreCase(charset, "X-UTF-32BE")) {
    return CharsetType::kUtf32BE;
  }
  if (equalsIgnoreCase(charset, "UTF-32LE") ||
      equalsIgnoreCase(charset, "UTF32LE") ||
      equalsIgnoreCase(charset, "UTF_32LE") ||
      equalsIgnoreCase(charset, "X-UTF-32LE")) {
    return CharsetType::kUtf32LE;
  }
  if (equalsIgnoreCase(charset, "UTF32") ||
      equalsIgnoreCase(charset, "UTF_32")) {
    return CharsetType::kUtf32;
  }
  if (isJis0208(charset)) {
    return CharsetType::kJis0208;
  }
  if (isJis0212(charset)) {
    return CharsetType::kJis0212;
  }

  UErrorCode error{U_ZERO_ERROR};
  const std::string charsetName{charset.data(), charset.size()};
  // Java's Charset.forName rejects names containing a NUL, whereas passing the
  // name through c_str() below would silently truncate at the first NUL and
  // resolve a different charset.
  if (charsetName.find('\0') != std::string::npos) {
    return CharsetType::kUnsupported;
  }
  std::unique_ptr<UConverter, decltype(&ucnv_close)> converter{
      ucnv_open(charsetName.c_str(), &error), &ucnv_close};
  if (U_FAILURE(error)) {
    return CharsetType::kUnsupported;
  }
  UErrorCode nameError{U_ZERO_ERROR};
  const char* canonicalName = ucnv_getName(converter.get(), &nameError);
  if (U_FAILURE(nameError) || canonicalName == nullptr ||
      !isJavaCharset(charset, canonicalName)) {
    return CharsetType::kUnsupported;
  }
  return CharsetType::kLegacy;
}

Status encode(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    CharsetType type,
    bool legacyCodingErrorAction) {
  switch (type) {
    case CharsetType::kUtf8:
      return encodeUtf8(result, input);
    case CharsetType::kUsAscii:
      return encodeSingleByte(
          result, input, charset, 0x80, legacyCodingErrorAction);
    case CharsetType::kIso8859_1:
      return encodeSingleByte(
          result, input, charset, 0x100, legacyCodingErrorAction);
    case CharsetType::kUtf16:
      return encodeUtf16(result, input, true, true);
    case CharsetType::kUtf16BE:
      return encodeUtf16(result, input, true, false);
    case CharsetType::kUtf16LE:
      return encodeUtf16(result, input, false, false);
    case CharsetType::kUtf16LEWithBom:
      return encodeUtf16(result, input, false, true);
    case CharsetType::kUtf32:
    case CharsetType::kUtf32BE:
      return encodeUtf32(result, input, true);
    case CharsetType::kUtf32LE:
      return encodeUtf32(result, input, false);
    case CharsetType::kJis0208:
      return encodeJisDoubleByte(
          result,
          input,
          charset,
          JisDoubleBytePlane::k0208,
          legacyCodingErrorAction);
    case CharsetType::kJis0212:
      return encodeJisDoubleByte(
          result,
          input,
          charset,
          JisDoubleBytePlane::k0212,
          legacyCodingErrorAction);
    case CharsetType::kLegacy:
      return encodeLegacy(result, input, charset, legacyCodingErrorAction);
    case CharsetType::kUnsupported:
      VELOX_UNREACHABLE();
  }
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::functions::sparksql::detail
