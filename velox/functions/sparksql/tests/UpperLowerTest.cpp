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

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class UpperLowerTest : public SparkFunctionBaseTest {
 protected:
  std::optional<std::string> lower(std::optional<std::string> str) {
    return evaluateOnce<std::string>("lower(c0)", str);
  }

  std::optional<std::string> upper(std::optional<std::string> str) {
    return evaluateOnce<std::string>("upper(c0)", str);
  }
};

TEST_F(UpperLowerTest, lowerAscii) {
  EXPECT_EQ("abcdefg", lower("ABCDEFG"));
  EXPECT_EQ("abcdefg", lower("abcdefg"));
  EXPECT_EQ("a b c d e f g", lower("a B c D e F g"));
}

TEST_F(UpperLowerTest, lowerUnicode) {
  EXPECT_EQ(
      "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ",
      lower("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"));
  EXPECT_EQ("αβγδεζηθικλμνξοπρσστυφχψ", lower("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΣΤΥΦΧΨ"));
  EXPECT_EQ(
      "абвгдежзийклмнопрстуфхцчшщъыьэюя",
      lower("АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"));
  EXPECT_EQ("i\xCC\x87", lower("\u0130"));
  EXPECT_EQ("i\xCC\x87", lower("I\xCC\x87"));
  EXPECT_EQ("\u010B", lower("\u010A"));
  EXPECT_EQ("\u0117", lower("\u0116"));
  EXPECT_EQ("\u0121", lower("\u0120"));
  EXPECT_EQ("\u017C", lower("\u017B"));
  EXPECT_EQ("\u0227", lower("\u0226"));
  EXPECT_EQ("\u022F", lower("\u022E"));
  EXPECT_EQ("\u1E03", lower("\u1E02"));
  EXPECT_EQ("\u1E0B", lower("\u1E0A"));
  EXPECT_EQ("\u1E1F", lower("\u1E1E"));
  EXPECT_EQ("\u1E23", lower("\u1E22"));
  EXPECT_EQ("\u1E41", lower("\u1E40"));
  EXPECT_EQ("\u1E45", lower("\u1E44"));
  EXPECT_EQ("\u1E57", lower("\u1E56"));
  EXPECT_EQ("\u1E59", lower("\u1E58"));
  EXPECT_EQ("\u1E61", lower("\u1E60"));
  EXPECT_EQ("\u1E65", lower("\u1E64"));
  EXPECT_EQ("\u1E67", lower("\u1E66"));
  EXPECT_EQ("\u1E69", lower("\u1E68"));
  EXPECT_EQ("\u1E6B", lower("\u1E6A"));
  EXPECT_EQ("\u1E87", lower("\u1E86"));
  EXPECT_EQ("\u1E8B", lower("\u1E8A"));
  EXPECT_EQ("\u1E8F", lower("\u1E8E"));
}

TEST_F(UpperLowerTest, lowerGreek) {
  // Basic cases: Σ at word end (preceded by cased letter, not followed by
  // cased letter) → ς
  EXPECT_EQ("πας", lower("ΠΑΣ"));
  EXPECT_EQ("πας ", lower("ΠΑΣ "));
  EXPECT_EQ("πασα", lower("ΠΑΣΑ"));

  // Case-ignorable characters (. is MidNumLet) don't break the context.
  // Σ preceded by cased Α, followed by case-ignorable '.', no cased after → ς
  EXPECT_EQ("πας.", lower("ΠΑΣ."));

  // Space is NOT case-ignorable, it breaks the context.
  // After space, 'A' is cased but space already stopped the forward scan.
  // Σ preceded by cased Α, space stops forward scan (no cased found) → ς
  EXPECT_EQ("πας   a", lower("ΠΑΣ   A"));

  // ` (backtick) is NOT case-ignorable, stops the forward scan.
  EXPECT_EQ("πας`", lower("ΠΑΣ`"));

  // Hebrew א and Japanese あ are NOT cased and NOT case-ignorable.
  // They stop the forward scan, no cased after → ς
  EXPECT_EQ("παςא", lower("ΠΑΣא"));
  EXPECT_EQ("παςあ", lower("ΠΑΣあ"));

  // ʰ (U+02B0) is a modifier letter (Lm), which is case-ignorable.
  // But it's also considered cased in our isCased() function.
  // So Σ is followed by cased character → σ
  EXPECT_EQ("πασʰ", lower("ΠΑΣʰ"));

  // ǅ (U+01C5) is a titlecase letter (Lt), which is cased.
  // Σ is followed by cased character → σ
  EXPECT_EQ("πασǆ", lower("ΠΑΣǅ"));

  // σ is lowercase letter (Ll), which is cased.
  // Σ is followed by cased character → σ
  EXPECT_EQ("πασσ", lower("ΠΑΣσ"));

  // Д is Cyrillic uppercase (Lu), which is cased.
  // Σ is followed by cased character → σ
  EXPECT_EQ("πασд", lower("ΠΑΣД"));

  // Cases where Σ is NOT preceded by a cased character → always σ
  // Space before Σ breaks the backward scan.
  EXPECT_EQ("hello σ", lower("hello Σ"));
  EXPECT_EQ("hello σ world", lower("hello Σ world"));
  EXPECT_EQ("ab σ", lower("ab Σ"));
  EXPECT_EQ("   σ", lower("   Σ"));

  // Standalone Σ (no cased before) → σ
  EXPECT_EQ("σ", lower("Σ"));

  // CJK characters are NOT cased and NOT case-ignorable.
  // They break the backward scan.
  EXPECT_EQ("中文σ", lower("中文Σ"));
  EXPECT_EQ("ab中σ中", lower("ab中Σ中"));

  // Σ preceded by cased, followed by CJK (stops forward scan) → ς
  EXPECT_EQ("abς中", lower("abΣ中"));

  // Σ preceded by cased, followed by symbol (stops forward scan) → ς
  EXPECT_EQ("abς<", lower("abΣ<"));

  // Apostrophe (') is case-ignorable (Single_Quote).
  // Σ preceded by cased 'o' (skipping '), not followed by cased → ς
  EXPECT_EQ("hello'ς", lower("hello'Σ"));

  // Dot (.) is case-ignorable (MidNumLet).
  EXPECT_EQ("hello.ς", lower("hello.Σ"));

  // Combining acute accent (U+0301) is case-ignorable (Mn category).
  // a + combining accent is still 'a' (cased).
  EXPECT_EQ("a\u0301ς", lower("a\u0301Σ"));

  // Multiple Σ in sequence.
  // First Σ: preceded by nothing → σ
  // Second Σ: preceded by cased Σ, followed by cased Σ → σ
  // Third Σ: preceded by cased Σ, not followed by cased → ς
  EXPECT_EQ("σσς", lower("ΣΣΣ"));

  // aΣbΣ: first Σ preceded by a, followed by b (cased) → σ
  //       second Σ preceded by b, not followed by cased → ς
  EXPECT_EQ("aσbς", lower("aΣbΣ"));

  // helloΣ: Σ directly preceded by cased 'o' → ς
  EXPECT_EQ("helloς", lower("helloΣ"));

  // Complex case with Japanese parenthesis.
  // Σ preceded by CJK '音', which breaks backward scan → σ
  EXPECT_EQ(
      "一只可爱的小狐狸能发出多少种声音σ（・□・；）",
      lower("一只可爱的小狐狸能发出多少种声音Σ（・□・；）"));
}

TEST_F(UpperLowerTest, upperAscii) {
  EXPECT_EQ("ABCDEFG", upper("abcdefg"));
  EXPECT_EQ("ABCDEFG", upper("ABCDEFG"));
  EXPECT_EQ("A B C D E F G", upper("a B c D e F g"));
}

TEST_F(UpperLowerTest, upperUnicode) {
  EXPECT_EQ(
      "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ",
      upper("àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ"));
  EXPECT_EQ("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΣΤΥΦΧΨ", upper("αβγδεζηθικλμνξοπρςστυφχψ"));
  EXPECT_EQ(
      "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
      upper("абвгдежзийклмнопрстуфхцчшщъыьэюя"));
  EXPECT_EQ("\u0049", upper("\u0069"));
  EXPECT_EQ("I\xCC\x87", upper("i\xCC\x87"));
  EXPECT_EQ("\u010A", upper("\u010B"));
  EXPECT_EQ("\u0116", upper("\u0117"));
  EXPECT_EQ("\u0120", upper("\u0121"));
  EXPECT_EQ("\u017B", upper("\u017C"));
  EXPECT_EQ("\u0226", upper("\u0227"));
  EXPECT_EQ("\u022E", upper("\u022F"));
  EXPECT_EQ("\u1E02", upper("\u1E03"));
  EXPECT_EQ("\u1E0A", upper("\u1E0B"));
  EXPECT_EQ("\u1E1E", upper("\u1E1F"));
  EXPECT_EQ("\u1E22", upper("\u1E23"));
  EXPECT_EQ("\u1E40", upper("\u1E41"));
  EXPECT_EQ("\u1E44", upper("\u1E45"));
  EXPECT_EQ("\u1E56", upper("\u1E57"));
  EXPECT_EQ("\u1E58", upper("\u1E59"));
  EXPECT_EQ("\u1E60", upper("\u1E61"));
  EXPECT_EQ("\u1E64", upper("\u1E65"));
  EXPECT_EQ("\u1E66", upper("\u1E67"));
  EXPECT_EQ("\u1E68", upper("\u1E69"));
  EXPECT_EQ("\u1E6A", upper("\u1E6B"));
  EXPECT_EQ("\u1E86", upper("\u1E87"));
  EXPECT_EQ("\u1E8A", upper("\u1E8B"));
  EXPECT_EQ("\u1E8E", upper("\u1E8F"));
}

TEST_F(UpperLowerTest, upperGreek) {
  EXPECT_EQ("ΠΑΣ", upper("πασ"));
  EXPECT_EQ("ΠΑΣ ", upper("πασ "));
  EXPECT_EQ("ΠΑΣA", upper("πασa"));
  EXPECT_EQ("ΠΑΣ", upper("πας"));
  EXPECT_EQ("ΠΑΣ ", upper("πας "));
  EXPECT_EQ("ΠΑΣA", upper("παςa"));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
