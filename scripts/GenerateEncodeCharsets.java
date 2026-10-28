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

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/** Generates the explicit Spark encode allowlist and encoder mappings using JDK 17. */
public final class GenerateEncodeCharsets {
  private static final String[] STANDARD = {
    "UTF-8", "US-ASCII", "ISO-8859-1", "UTF-16", "UTF-16BE", "UTF-16LE", "UTF-32"
  };
  private static final String[] LEGACY = {
    "IBM037", "IBM437",
    "ISO-8859-2", "ISO-8859-3", "ISO-8859-4", "ISO-8859-5", "ISO-8859-6",
    "ISO-8859-7", "ISO-8859-8", "ISO-8859-9", "x-iso-8859-11",
    "ISO-8859-13", "ISO-8859-15", "ISO-8859-16", "KOI8-R", "KOI8-U",
    "windows-1250", "windows-1251", "windows-1252", "windows-1253",
    "windows-1254", "windows-1255", "windows-1256", "windows-1257", "windows-1258"
  };

  private record Mapping(int codePoint, int encodedByte) {}

  private record Encoding(
      Charset charset, int[] low, List<Mapping> extra, int replacement, boolean asciiCompatible) {}

  private static Encoding inspect(String name) {
    Charset charset = Charset.forName(name);
    if (!charset.canEncode()) {
      throw new IllegalStateException("Decode-only charset: " + name);
    }
    CharsetEncoder encoder = charset.newEncoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT);
    if (encoder.maxBytesPerChar() != 1 || encoder.replacement().length != 1) {
      throw new IllegalStateException("Not a single-byte encoder: " + name);
    }
    int[] low = new int[256];
    Arrays.fill(low, 256);
    List<Mapping> extra = new ArrayList<>();
    CharBuffer input = CharBuffer.allocate(2);
    ByteBuffer output = ByteBuffer.allocate(16);
    for (int codePoint = 0; codePoint <= Character.MAX_CODE_POINT; ++codePoint) {
      if (codePoint >= Character.MIN_SURROGATE && codePoint <= Character.MAX_SURROGATE) {
        continue;
      }
      input.clear();
      if (Character.isBmpCodePoint(codePoint)) {
        input.put((char) codePoint);
      } else {
        input.put(Character.highSurrogate(codePoint));
        input.put(Character.lowSurrogate(codePoint));
      }
      input.flip();
      output.clear();
      encoder.reset();
      CoderResult result = encoder.encode(input, output, true);
      if (result.isError()) {
        continue;
      }
      if (result.isOverflow() || input.hasRemaining()) {
        throw new IllegalStateException("Unexpected encoder overflow: " + name);
      }
      result = encoder.flush(output);
      if (!result.isUnderflow() || output.position() != 1) {
        throw new IllegalStateException("Not a scalar single-byte mapping: " + name);
      }
      int encodedByte = output.get(0) & 255;
      if (codePoint < 256) {
        low[codePoint] = encodedByte;
      } else {
        extra.add(new Mapping(codePoint, encodedByte));
      }
    }
    boolean asciiCompatible = true;
    for (int codePoint = 0; codePoint < 128; ++codePoint) {
      asciiCompatible &= low[codePoint] == codePoint;
    }
    System.err.printf(
        "%s: checked every Unicode scalar, %d extra mappings, replacement=%02X, ASCII=%s%n",
        charset.name(), extra.size(), encoder.replacement()[0] & 255, asciiCompatible);
    return new Encoding(charset, low, extra, encoder.replacement()[0] & 255, asciiCompatible);
  }

  private static Map<String, Integer> aliases(List<Charset> charsets) {
    Map<String, Integer> result = new TreeMap<>();
    for (int index = 0; index < charsets.size(); ++index) {
      Charset charset = charsets.get(index);
      List<String> names = new ArrayList<>(charset.aliases());
      names.add(charset.name());
      for (String name : names) {
        if (!Charset.forName(name).equals(charset)
            || name.chars().anyMatch(value -> value > 127)) {
          throw new IllegalStateException("Unverified alias: " + name);
        }
        String folded = name.toUpperCase(Locale.ROOT);
        Integer previous = result.putIfAbsent(folded, index);
        if (previous != null && previous != index) {
          throw new IllegalStateException("Ambiguous alias: " + name);
        }
      }
    }
    return result;
  }

  private static void printAliases(String name, Map<String, Integer> aliases) {
    System.out.printf(
        "inline constexpr std::array<Alias, %d> %s{{%n", aliases.size(), name);
    aliases.forEach((alias, index) -> System.out.printf("  {\"%s\", %d},%n", alias, index));
    System.out.println("}};");
  }

  public static void main(String[] args) {
    if (Runtime.version().feature() != 17 || args.length != 0) {
      throw new IllegalArgumentException("Run with JDK 17 and no arguments");
    }
    List<Encoding> encodings = Arrays.stream(LEGACY).map(GenerateEncodeCharsets::inspect).toList();
    Map<String, Integer> standardAliases =
        aliases(Arrays.stream(STANDARD).map(Charset::forName).toList());
    Map<String, Integer> legacyAliases =
        aliases(encodings.stream().map(Encoding::charset).toList());
    System.out.println("""
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

        #include <array>
        #include <cstdint>
        #include <string_view>

        // Generated with JDK 17; do not edit.
        // Regenerate: java scripts/GenerateEncodeCharsets.java | clang-format
        namespace facebook::velox::functions::sparksql::detail::single_byte {

        /// Stores an encoder-only or non-Latin-1 Unicode mapping.
        struct Mapping {
          /// Identifies the input Unicode scalar.
          char32_t codePoint;
          /// Preserves the encoder output, including a zero byte.
          uint8_t encodedByte;
        };

        /// Describes a verified single-byte encoder and its replacement.
        struct Charset {
          /// Identifies the canonical JDK 17 charset.
          std::string_view name;
          /// Locates this charset's sorted span in kExtraMappings.
          uint16_t extraOffset;
          /// Bounds this charset's span in the shared mapping table.
          uint16_t extraSize;
          /// Preserves the encoder's default replacement byte.
          uint8_t replacement;
          /// Holds only when all 128 ASCII scalars encode identically.
          bool asciiCompatible;
        };

        /// Resolves an explicitly verified, ASCII-folded Java charset name.
        struct Alias {
          /// Stores the exact name with ASCII letters folded to uppercase.
          std::string_view name;
          /// Indexes the corresponding standard or legacy charset table.
          uint8_t charsetIndex;
        };

        /// Distinguishes an unmappable scalar from every possible output byte.
        inline constexpr uint16_t kUnmappable = 256;
        """);
    System.out.println("/// Maps Latin-1 scalars directly; kUnmappable marks absent mappings.");
    System.out.printf(
        "inline constexpr std::array<std::array<uint16_t, 256>, %d> kLowMappings{{%n",
        encodings.size());
    for (Encoding encoding : encodings) {
      System.out.println("  {{");
      for (int index = 0; index < 256; ++index) {
        if (index % 16 == 0) {
          System.out.print("    ");
        }
        System.out.printf("%d,%s", encoding.low()[index], index % 16 == 15 ? "\n" : " ");
      }
      System.out.println("  }},");
    }
    System.out.println("}};");
    int numExtra = encodings.stream().mapToInt(encoding -> encoding.extra().size()).sum();
    if (numExtra > 65535) {
      throw new IllegalStateException("Mapping offsets exceed uint16_t");
    }
    System.out.printf("inline constexpr std::array<Mapping, %d> kExtraMappings{{%n", numExtra);
    for (Encoding encoding : encodings) {
      for (Mapping mapping : encoding.extra()) {
        System.out.printf("  {0x%X, 0x%02X},%n", mapping.codePoint(), mapping.encodedByte());
      }
    }
    System.out.println("}};");
    System.out.printf("inline constexpr std::array<Charset, %d> kCharsets{{%n", encodings.size());
    int offset = 0;
    for (Encoding encoding : encodings) {
      System.out.printf("  {\"%s\", %d, %d, 0x%02X, %s},%n",
          encoding.charset().name(), offset, encoding.extra().size(),
          encoding.replacement(), encoding.asciiCompatible());
      offset += encoding.extra().size();
    }
    System.out.println("}};");
    printAliases("kStandardAliases", standardAliases);
    printAliases("kLegacyAliases", legacyAliases);
    System.out.println("} // namespace facebook::velox::functions::sparksql::detail::single_byte");
  }
}
