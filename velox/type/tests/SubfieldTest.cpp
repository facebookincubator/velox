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
#include "velox/type/Subfield.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/type/Tokenizer.h"

using namespace facebook::velox::common;

std::vector<std::unique_ptr<Subfield::PathElement>> tokenize(
    const std::string& path,
    std::shared_ptr<const Separators> separators = Separators::get()) {
  std::vector<std::unique_ptr<Subfield::PathElement>> elements;
  Tokenizer tokenizer(path, std::move(separators));
  while (tokenizer.hasNext()) {
    elements.push_back(tokenizer.next());
  }
  return elements;
}

void assertInvalidSubfield(
    const std::string& subfield,
    const std::string& message) {
  try {
    tokenize(subfield);
    ASSERT_TRUE(false) << "Expected an exception parsing " << subfield;
  } catch (facebook::velox::VeloxRuntimeError& e) {
    ASSERT_EQ(e.message(), message);
  }
}

TEST(SubfieldTest, invalidPaths) {
  assertInvalidSubfield("a[b]", "Invalid index b]");
  assertInvalidSubfield("a[2", "Invalid subfield path: a[2^");
  assertInvalidSubfield("a.*", "Invalid subfield path: a.^*");
  assertInvalidSubfield("a[2].[3].", "Invalid subfield path: a[2].^[3].");
}

void testColumnName(
    const std::string& name,
    std::shared_ptr<const Separators> separators = Separators::get()) {
  auto elements = tokenize(name, std::move(separators));
  EXPECT_EQ(elements.size(), 1);
  EXPECT_EQ(*elements[0].get(), Subfield::NestedField(name));
}

TEST(SubfieldTest, columnNamesWithSpecialCharacters) {
  testColumnName("two words");
  testColumnName("two  words");
  testColumnName("one two three");
  testColumnName("$bucket");
  testColumnName("apollo-11");
  testColumnName("a/b/c:12");
  testColumnName("@basis");
  testColumnName("@basis|city_id");
  auto separators = std::make_shared<Separators>();
  separators->dot = '\0';
  testColumnName("city.id@address:number/date|day$a-b$10_bucket", separators);
}

std::vector<std::unique_ptr<Subfield::PathElement>> createElements() {
  std::vector<std::unique_ptr<Subfield::PathElement>> elements;
  elements.push_back(std::make_unique<Subfield::NestedField>("b"));
  elements.push_back(std::make_unique<Subfield::LongSubscript>(2));
  elements.push_back(std::make_unique<Subfield::LongSubscript>(-1));
  elements.push_back(std::make_unique<Subfield::StringSubscript>("z"));
  elements.push_back(std::make_unique<Subfield::AllSubscripts>());
  elements.push_back(std::make_unique<Subfield::StringSubscript>("34"));
  elements.push_back(std::make_unique<Subfield::StringSubscript>("b \"test\""));
  elements.push_back(std::make_unique<Subfield::StringSubscript>("\"abc"));
  elements.push_back(std::make_unique<Subfield::StringSubscript>("abc\""));
  elements.push_back(std::make_unique<Subfield::StringSubscript>("ab\"cde"));
  return elements;
}

void testRoundTrip(const Subfield& path) {
  auto actual = Subfield(tokenize(path.toString()));
  ASSERT_TRUE(actual.valid());
  EXPECT_EQ(actual, path) << "at " << path.toString() << ", "
                          << actual.toString();
}

TEST(SubfieldTest, basic) {
  auto elements = createElements();
  for (auto& element : elements) {
    std::vector<std::unique_ptr<Subfield::PathElement>> newElements;
    newElements.push_back(std::make_unique<Subfield::NestedField>("a"));
    newElements.push_back(element->clone());
    testRoundTrip(Subfield(std::move(newElements)));
  }

  for (auto& element : elements) {
    for (auto& secondElement : elements) {
      std::vector<std::unique_ptr<Subfield::PathElement>> newElements;
      newElements.push_back(std::make_unique<Subfield::NestedField>("a"));
      newElements.push_back(element->clone());
      newElements.push_back(secondElement->clone());
      testRoundTrip(Subfield(std::move(newElements)));
    }
  }

  for (auto& element : elements) {
    for (auto& secondElement : elements) {
      for (auto& thirdElement : elements) {
        std::vector<std::unique_ptr<Subfield::PathElement>> newElements;
        newElements.push_back(std::make_unique<Subfield::NestedField>("a"));
        newElements.push_back(element->clone());
        newElements.push_back(secondElement->clone());
        newElements.push_back(thirdElement->clone());
        testRoundTrip(Subfield(std::move(newElements)));
      }
    }
  }

  ASSERT_FALSE(Subfield().valid());
  ASSERT_EQ(Subfield().toString(), "");
}

TEST(SubfieldTest, prefix) {
  EXPECT_FALSE(Subfield("a").isPrefix(Subfield("a")));
  EXPECT_TRUE(Subfield("a.b").isPrefix(Subfield("a.b.c")));
  EXPECT_TRUE(Subfield("a.b").isPrefix(Subfield("a.b[1]")));
  EXPECT_TRUE(Subfield("a.b").isPrefix(Subfield("a.b[\"d\"]")));
  EXPECT_FALSE(Subfield("a.c").isPrefix(Subfield("a.b.c")));
  EXPECT_FALSE(Subfield("a.b.c").isPrefix(Subfield("a.b")));
}

TEST(SubfieldTest, hash) {
  std::unordered_set<Subfield> subfields;
  subfields.emplace("a.b");
  subfields.emplace("a[\"b\"]");
  subfields.emplace("a.b.c");
  EXPECT_EQ(subfields.size(), 3);
  EXPECT_TRUE(subfields.find(Subfield("a.b")) != subfields.end());
  subfields.emplace("a.b.c");
  subfields.emplace("a[\"b\"]");
  EXPECT_EQ(subfields.size(), 3);
}

TEST(SubfieldTest, longSubscript) {
  Subfield subfield("a[3309189884973035076]");
  ASSERT_EQ(subfield.path().size(), 2);
  auto* longSubscript =
      dynamic_cast<const Subfield::LongSubscript*>(subfield.path()[1].get());
  ASSERT_TRUE(longSubscript);
  ASSERT_EQ(longSubscript->index(), 3309189884973035076);
}

TEST(SubfieldTest, cardinalityOnlySubscript) {
  // Test basic parsing of x[$]
  Subfield subfield("x[$]");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  // Verify toString round-trip
  EXPECT_EQ(subfield.toString(), "x[$]");

  // Test with nested field
  Subfield nestedSubfield("a.b[$]");
  ASSERT_EQ(nestedSubfield.path().size(), 3);
  EXPECT_EQ(nestedSubfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(nestedSubfield.path()[1]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(
      nestedSubfield.path()[2]->kind(), SubfieldKind::kArrayOrMapSubscript);
  EXPECT_EQ(nestedSubfield.toString(), "a.b[$]");
}

TEST(SubfieldTest, cardinalityOnlyEquality) {
  // Test equality of cardinality-only subscripts
  Subfield s1("x[$]");
  Subfield s2("x[$]");
  Subfield s3("y[$]");

  EXPECT_EQ(s1, s2);
  EXPECT_NE(s1, s3);

  // Test path element equality
  auto cardinalityOnly1 =
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr);
  auto cardinalityOnly2 =
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr);
  EXPECT_EQ(*cardinalityOnly1, *cardinalityOnly2);
}

TEST(SubfieldTest, cardinalityOnlyHash) {
  // Test hashing works correctly
  std::unordered_set<Subfield> subfields;
  subfields.emplace("x[$]");
  subfields.emplace("y[$]");
  subfields.emplace("x[$]"); // Duplicate, should not increase size

  EXPECT_EQ(subfields.size(), 2);
  EXPECT_TRUE(subfields.find(Subfield("x[$]")) != subfields.end());
  EXPECT_TRUE(subfields.find(Subfield("y[$]")) != subfields.end());
}

TEST(SubfieldTest, nestedCardinalityOnly) {
  // Test arr[*][$] for array of maps
  Subfield subfield("arr[*][$]");
  ASSERT_EQ(subfield.path().size(), 3);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kAllSubscripts);
  EXPECT_EQ(subfield.path()[2]->kind(), SubfieldKind::kArrayOrMapSubscript);
  EXPECT_EQ(subfield.toString(), "arr[*][$]");

  // Test map_col[*][*][$] for nested structures
  Subfield complexSubfield("data[*][\"key\"][$]");
  ASSERT_EQ(complexSubfield.path().size(), 4);
  EXPECT_EQ(complexSubfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(complexSubfield.path()[1]->kind(), SubfieldKind::kAllSubscripts);
  EXPECT_EQ(complexSubfield.path()[2]->kind(), SubfieldKind::kStringSubscript);
  EXPECT_EQ(
      complexSubfield.path()[3]->kind(), SubfieldKind::kArrayOrMapSubscript);
}

TEST(SubfieldTest, cardinalityOnlyRoundTrip) {
  // Test round-trip through tokenization
  std::vector<std::string> testPaths = {
      "x[$]",
      "a.b[$]",
      "arr[*][$]",
      "map_col[\"key\"][$]",
      "data[123][$]",
      "nested[*][\"field\"][$]"};

  for (const auto& path : testPaths) {
    Subfield original(path);
    ASSERT_TRUE(original.valid());

    // Convert to string and parse back
    auto roundTrip = Subfield(original.toString());
    ASSERT_TRUE(roundTrip.valid());
    EXPECT_EQ(original, roundTrip) << "Failed round-trip for: " << path;
    EXPECT_EQ(original.toString(), roundTrip.toString());
  }
}

TEST(SubfieldTest, cardinalityOnlyInCreateElements) {
  // Add ArrayOrMapSubscript to the mix of all subscript types
  auto elements = createElements();
  elements.push_back(
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr));

  // Test with base field
  std::vector<std::unique_ptr<Subfield::PathElement>> newElements;
  newElements.push_back(std::make_unique<Subfield::NestedField>("a"));
  newElements.push_back(
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr));
  testRoundTrip(Subfield(std::move(newElements)));
}

TEST(SubfieldTest, arrayOrMapSubscriptCardinalityOnly) {
  // Test basic ArrayOrMapSubscript in cardinality-only mode (equivalent to [$])
  auto arrayOrMap =
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr);
  EXPECT_TRUE(arrayOrMap->isCardinalityOnly());
  EXPECT_FALSE(arrayOrMap->includeKeys());
  EXPECT_FALSE(arrayOrMap->includeValues());
  EXPECT_EQ(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->toString(), "[$]");

  // Test in a full subfield path
  std::vector<std::unique_ptr<Subfield::PathElement>> elements;
  elements.push_back(std::make_unique<Subfield::NestedField>("x"));
  elements.push_back(
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr));
  Subfield subfield(std::move(elements));
  EXPECT_EQ(subfield.toString(), "x[$]");
}

TEST(SubfieldTest, arrayOrMapSubscriptEquality) {
  // Test equality of cardinality-only subscripts
  auto sub1 =
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr);
  auto sub2 =
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr);
  EXPECT_EQ(*sub1, *sub2);

  // Test clone
  auto cloned = sub1->clone();
  EXPECT_EQ(*sub1, *cloned);
}

TEST(SubfieldTest, arrayOrMapSubscriptDesignPatterns) {
  // Pattern 1: cardinality-only mode [$]
  auto cardinalityOnly =
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr);
  EXPECT_TRUE(cardinalityOnly->isCardinalityOnly());
  EXPECT_EQ(cardinalityOnly->toString(), "[$]");

  // Future patterns (not yet implemented in parser, but data structure
  // supports): Pattern 2: All keys only [K*, *]
  auto allKeys = std::make_unique<Subfield::ArrayOrMapSubscript>(
      true /* includeKeys */,
      false /* includeValues */,
      std::make_unique<Subfield::AllSubscripts>());
  EXPECT_TRUE(allKeys->includeKeys());
  EXPECT_FALSE(allKeys->includeValues());
  EXPECT_NE(allKeys->subscript(), nullptr);
  EXPECT_EQ(allKeys->subscript()->kind(), SubfieldKind::kAllSubscripts);

  // Pattern 3: Only keys with LongSubscript 42 [K*, 42]
  auto keysWithLong = std::make_unique<Subfield::ArrayOrMapSubscript>(
      true /* includeKeys */,
      false /* includeValues */,
      std::make_unique<Subfield::LongSubscript>(42));
  EXPECT_TRUE(keysWithLong->includeKeys());
  EXPECT_FALSE(keysWithLong->includeValues());
  EXPECT_EQ(keysWithLong->subscript()->kind(), SubfieldKind::kLongSubscript);
  EXPECT_EQ(keysWithLong->toString(), "[K*, 42]");

  // Pattern 4: Only keys with StringSubscript 'foo' [K*, 'foo']
  auto keysWithString = std::make_unique<Subfield::ArrayOrMapSubscript>(
      true /* includeKeys */,
      false /* includeValues */,
      std::make_unique<Subfield::StringSubscript>("foo"));
  EXPECT_TRUE(keysWithString->includeKeys());
  EXPECT_FALSE(keysWithString->includeValues());
  EXPECT_EQ(
      keysWithString->subscript()->kind(), SubfieldKind::kStringSubscript);
  EXPECT_EQ(keysWithString->toString(), "[K*, 'foo']");

  // Pattern 5: All values only [V*, *]
  auto allValues = std::make_unique<Subfield::ArrayOrMapSubscript>(
      false /* includeKeys */,
      true /* includeValues */,
      std::make_unique<Subfield::AllSubscripts>());
  EXPECT_FALSE(allValues->includeKeys());
  EXPECT_TRUE(allValues->includeValues());
  EXPECT_EQ(allValues->subscript()->kind(), SubfieldKind::kAllSubscripts);

  // Pattern 6: Only values with key equals to LongSubscript 42 [V*, 42]
  auto valuesWithLong = std::make_unique<Subfield::ArrayOrMapSubscript>(
      false /* includeKeys */,
      true /* includeValues */,
      std::make_unique<Subfield::LongSubscript>(42));
  EXPECT_FALSE(valuesWithLong->includeKeys());
  EXPECT_TRUE(valuesWithLong->includeValues());
  EXPECT_EQ(valuesWithLong->subscript()->kind(), SubfieldKind::kLongSubscript);
  EXPECT_EQ(valuesWithLong->toString(), "[V*, 42]");

  // Pattern 7: Only values with key equals to StringSubscript 'foo' [V*, 'foo']
  auto valuesWithString = std::make_unique<Subfield::ArrayOrMapSubscript>(
      false /* includeKeys */,
      true /* includeValues */,
      std::make_unique<Subfield::StringSubscript>("foo"));
  EXPECT_FALSE(valuesWithString->includeKeys());
  EXPECT_TRUE(valuesWithString->includeValues());
  EXPECT_EQ(
      valuesWithString->subscript()->kind(), SubfieldKind::kStringSubscript);
  EXPECT_EQ(valuesWithString->toString(), "[V*, 'foo']");

  // Pattern 8: All keys and values with no [*, *]
  auto allKeysAndValues = std::make_unique<Subfield::ArrayOrMapSubscript>(
      true /* includeKeys */,
      true /* includeValues */,
      std::make_unique<Subfield::AllSubscripts>());
  VELOX_ASSERT_THROW(
      allKeysAndValues->toString(),
      "Invalid subfield pushdown, should use kAllSubscripts, LongSubscript or StringSubscript directly");

  // Pattern 9: All keys and values with StringSubscript
  auto keysAndValuesWithString =
      std::make_unique<Subfield::ArrayOrMapSubscript>(
          true /* includeKeys */,
          true /* includeValues */,
          std::make_unique<Subfield::StringSubscript>("foo"));
  VELOX_ASSERT_THROW(
      keysAndValuesWithString->toString(),
      "Invalid subfield pushdown, should use kAllSubscripts, LongSubscript or StringSubscript directly");

  // Pattern 10: All keys and values with LongSubscript
  auto keysAndValuesWithLong = std::make_unique<Subfield::ArrayOrMapSubscript>(
      true /* includeKeys */,
      true /* includeValues */,
      std::make_unique<Subfield::LongSubscript>(42));
  VELOX_ASSERT_THROW(
      keysAndValuesWithLong->toString(),
      "Invalid subfield pushdown, should use kAllSubscripts, LongSubscript or StringSubscript directly");
}

TEST(SubfieldTest, arrayOrMapSubscriptHash) {
  // Test hashing for cardinality-only mode
  auto sub1 =
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr);
  auto sub2 =
      std::make_unique<Subfield::ArrayOrMapSubscript>(false, false, nullptr);
  EXPECT_EQ(sub1->hash(), sub2->hash());

  // Test hashing for different configurations
  auto allKeys = std::make_unique<Subfield::ArrayOrMapSubscript>(
      true, false, std::make_unique<Subfield::AllSubscripts>());
  auto allValues = std::make_unique<Subfield::ArrayOrMapSubscript>(
      false, true, std::make_unique<Subfield::AllSubscripts>());
  // Different configurations should have different hashes (usually)
  EXPECT_NE(allKeys->hash(), allValues->hash());
}

TEST(SubfieldTest, mapKeysOnlyAllSubscripts) {
  // Test [K*, *] pattern - read all keys, skip values
  Subfield subfield("x[K*, *]");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      subfield.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_TRUE(arrayOrMap->includeKeys());
  EXPECT_FALSE(arrayOrMap->includeValues());
  EXPECT_EQ(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->toString(), "[K*, *]");
}

TEST(SubfieldTest, mapKeysOnlyWithLongSubscript) {
  // Test [K*, 42] pattern - read only keys matching index 42
  Subfield subfield("x[K*, 42]");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      subfield.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_TRUE(arrayOrMap->includeKeys());
  EXPECT_FALSE(arrayOrMap->includeValues());
  ASSERT_NE(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->subscript()->kind(), SubfieldKind::kLongSubscript);
  EXPECT_EQ(
      arrayOrMap->subscript()->as<Subfield::LongSubscript>()->index(), 42);
  EXPECT_EQ(arrayOrMap->toString(), "[K*, 42]");
}

TEST(SubfieldTest, mapKeysOnlyWithStringSubscript) {
  // Test [K*, 'foo'] pattern - read only keys matching 'foo'
  Subfield subfield("x[K*, 'foo']");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      subfield.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_TRUE(arrayOrMap->includeKeys());
  EXPECT_FALSE(arrayOrMap->includeValues());
  ASSERT_NE(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->subscript()->kind(), SubfieldKind::kStringSubscript);
  EXPECT_EQ(
      arrayOrMap->subscript()->as<Subfield::StringSubscript>()->index(), "foo");
  EXPECT_EQ(arrayOrMap->toString(), "[K*, 'foo']");
}

TEST(SubfieldTest, mapKeysOnlyWithDoubleQuotedStringSubscript) {
  // Test [K*, "foo"] pattern - read only keys matching "foo"
  Subfield subfield("x[K*, \"foo\"]");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      subfield.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_TRUE(arrayOrMap->includeKeys());
  EXPECT_FALSE(arrayOrMap->includeValues());
  ASSERT_NE(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->subscript()->kind(), SubfieldKind::kStringSubscript);
  EXPECT_EQ(
      arrayOrMap->subscript()->as<Subfield::StringSubscript>()->index(), "foo");
}

TEST(SubfieldTest, mapValuesOnlyAllSubscripts) {
  // Test [V*, *] pattern - read all values, skip keys
  Subfield subfield("x[V*, *]");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      subfield.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_FALSE(arrayOrMap->includeKeys());
  EXPECT_TRUE(arrayOrMap->includeValues());
  EXPECT_EQ(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->toString(), "[V*, *]");
}

TEST(SubfieldTest, mapValuesOnlyWithLongSubscript) {
  // Test [V*, 42] pattern - read only values where key matches index 42
  Subfield subfield("x[V*, 42]");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      subfield.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_FALSE(arrayOrMap->includeKeys());
  EXPECT_TRUE(arrayOrMap->includeValues());
  ASSERT_NE(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->subscript()->kind(), SubfieldKind::kLongSubscript);
  EXPECT_EQ(
      arrayOrMap->subscript()->as<Subfield::LongSubscript>()->index(), 42);
  EXPECT_EQ(arrayOrMap->toString(), "[V*, 42]");
}

TEST(SubfieldTest, mapValuesOnlyWithStringSubscript) {
  // Test [V*, 'foo'] pattern - read only values where key matches 'foo'
  Subfield subfield("x[V*, 'foo']");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      subfield.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_FALSE(arrayOrMap->includeKeys());
  EXPECT_TRUE(arrayOrMap->includeValues());
  ASSERT_NE(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->subscript()->kind(), SubfieldKind::kStringSubscript);
  EXPECT_EQ(
      arrayOrMap->subscript()->as<Subfield::StringSubscript>()->index(), "foo");
  EXPECT_EQ(arrayOrMap->toString(), "[V*, 'foo']");
}

TEST(SubfieldTest, mapValuesOnlyWithDoubleQuotedStringSubscript) {
  // Test [V*, "foo"] pattern - read only values where key matches "foo"
  Subfield subfield("x[V*, \"foo\"]");
  ASSERT_EQ(subfield.path().size(), 2);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      subfield.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_FALSE(arrayOrMap->includeKeys());
  EXPECT_TRUE(arrayOrMap->includeValues());
  ASSERT_NE(arrayOrMap->subscript(), nullptr);
  EXPECT_EQ(arrayOrMap->subscript()->kind(), SubfieldKind::kStringSubscript);
  EXPECT_EQ(
      arrayOrMap->subscript()->as<Subfield::StringSubscript>()->index(), "foo");
}

TEST(SubfieldTest, mapSubscriptWithWhitespace) {
  // Test patterns with whitespace after comma
  Subfield s1("x[K*,    *]"); // Multiple spaces
  ASSERT_EQ(s1.path().size(), 2);
  const auto* a1 = s1.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(a1, nullptr);
  EXPECT_TRUE(a1->includeKeys());

  Subfield s2("x[V*,42]"); // No space
  ASSERT_EQ(s2.path().size(), 2);
  const auto* a2 = s2.path()[1]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(a2, nullptr);
  EXPECT_TRUE(a2->includeValues());
}

TEST(SubfieldTest, nestedMapSubscript) {
  // Test nested paths with MAP subscripts
  Subfield subfield("a.b[K*, *]");
  ASSERT_EQ(subfield.path().size(), 3);
  EXPECT_EQ(subfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[1]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(subfield.path()[2]->kind(), SubfieldKind::kArrayOrMapSubscript);

  // Test with existing subscripts
  Subfield complexSubfield("arr[*][K*, 'key']");
  ASSERT_EQ(complexSubfield.path().size(), 3);
  EXPECT_EQ(complexSubfield.path()[0]->kind(), SubfieldKind::kNestedField);
  EXPECT_EQ(complexSubfield.path()[1]->kind(), SubfieldKind::kAllSubscripts);
  EXPECT_EQ(
      complexSubfield.path()[2]->kind(), SubfieldKind::kArrayOrMapSubscript);

  const auto* arrayOrMap =
      complexSubfield.path()[2]->as<Subfield::ArrayOrMapSubscript>();
  ASSERT_NE(arrayOrMap, nullptr);
  EXPECT_TRUE(arrayOrMap->includeKeys());
  EXPECT_FALSE(arrayOrMap->includeValues());
}
