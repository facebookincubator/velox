/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <folly/dynamic.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/type/OpaqueCustomTypes.h"
#include "velox/type/Variant.h"

namespace facebook::velox {
namespace {

class OpaqueCustomTypeTest : public testing::Test {
 public:
  OpaqueCustomTypeTest()
      : testCustomType_(TestCustomTypeRegister::VeloxType::get()) {}

 protected:
  struct TestCustomType {
    std::string name;

    bool operator==(const TestCustomType& other) const {
      return name == other.name;
    }

    folly::dynamic serialize() const {
      folly::dynamic obj = folly::dynamic::object();
      obj["name"] = name;
      return obj;
    }

    static std::shared_ptr<TestCustomType> create(const folly::dynamic& obj) {
      return std::make_shared<TestCustomType>(obj["name"].asString());
    }
  };

  static constexpr char testCustomTypeName[] = "my_test_custom_type";
  using TestCustomTypeRegister =
      OpaqueCustomTypeRegister<TestCustomType, testCustomTypeName>;

  TestCustomTypeRegister::VeloxTypePtr testCustomType_;

  void testUnregisterType() {
    TestCustomTypeRegister::unregisterType();
    ASSERT_EQ(getCustomType(testCustomTypeName, {}), nullptr);
    VELOX_ASSERT_THROW(
        testCustomType_->getSerializeFunc(),
        "No serialization function registered for my_test_custom_type");
    VELOX_ASSERT_THROW(
        testCustomType_->getDeserializeFunc(),
        "No deserialization function registered for my_test_custom_type");
  }

  void testTypeSerde() {
    auto serializedCustomType = testCustomType_->serialize();
    EXPECT_EQ(testCustomType_->name(), serializedCustomType["type"]);
    auto deserializedType = Type::create(serializedCustomType);
    EXPECT_EQ(testCustomType_, deserializedType);
  }
};

TEST_F(OpaqueCustomTypeTest, registerWithoutSerde) {
  // Register type without serialization/deserialization and build a TypePtr.
  TestCustomTypeRegister::registerType();
  testTypeSerde();
  testUnregisterType();
}

TEST_F(OpaqueCustomTypeTest, registerWithSerde) {
  // Register type with serialization/deserialization and build a TypePtr.
  TestCustomTypeRegister::registerType(
      /*serializeValueFunc=*/
      [](const auto& obj) { return folly::toJson(obj->serialize()); },
      /*deserializeValueFunc=*/
      [](const auto& str) {
        return TestCustomType::create(folly::parseJson(str));
      });

  // Test value serialization/deserialization, with serde functions provided in
  // TestCustomTypeRegister::registerType() method.
  auto original = std::make_shared<TestCustomType>("test");
  auto opaque = Variant::create(Variant::opaque(original).serialize())
                    .value<TypeKind::OPAQUE>()
                    .obj;
  auto copy = std::static_pointer_cast<TestCustomType>(opaque);
  EXPECT_EQ(*original, *copy);

  testTypeSerde();
  testUnregisterType();
}

TEST_F(OpaqueCustomTypeTest, registerWithIncompleteSerde) {
  VELOX_ASSERT_THROW(
      TestCustomTypeRegister::registerType(
          /*serializeValueFunc=*/nullptr,
          /*deserializeValueFunc=*/
          [](const auto& str) {
            return TestCustomType::create(folly::parseJson(str));
          }),
      "Both serialization and deserialization functions need to be registered for custom type my_test_custom_type");

  VELOX_ASSERT_THROW(
      TestCustomTypeRegister::registerType(
          /*serializeValueFunc=*/
          [](const auto& obj) { return folly::toJson(obj->serialize()); },
          /*deserializeValueFunc=*/nullptr),
      "Both serialization and deserialization functions need to be registered for custom type my_test_custom_type");

  testUnregisterType();
}

} // namespace
} // namespace facebook::velox
