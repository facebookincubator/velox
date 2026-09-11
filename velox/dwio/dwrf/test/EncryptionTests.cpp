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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "velox/dwio/common/encryption/TestProvider.h"
#include "velox/dwio/dwrf/test/OrcTest.h"
#include "velox/type/fbhive/HiveTypeParser.h"

using namespace facebook::velox::dwio;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::common::encryption;
using namespace facebook::velox::dwio::common::encryption::test;
using namespace facebook::velox::dwrf::encryption;
using namespace facebook::velox::type::fbhive;

TEST(Encryption, notEncrypted) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  TestEncrypterFactory factory;
  auto handler = EncryptionHandler::create(type, spec, &factory);
  ASSERT_FALSE(handler->isEncrypted());
}

TEST(Encryption, rootThenField) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  spec.withRootEncryptionProperties(
      std::make_shared<TestEncryptionProperties>("a"));
  ASSERT_THROW(
      spec.withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(0).withEncryptionProperties(
              std::make_shared<TestEncryptionProperties>("b"))),
      exception::LoggedException);
}

TEST(Encryption, fieldThenRoot) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  spec.withEncryptedField(
      FieldEncryptionSpecification{}.withIndex(0).withEncryptionProperties(
          std::make_shared<TestEncryptionProperties>("a")));
  ASSERT_THROW(
      spec.withRootEncryptionProperties(
          std::make_shared<TestEncryptionProperties>("b")),
      exception::LoggedException);
}

TEST(Encryption, root) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int, b:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  spec.withRootEncryptionProperties(
      std::make_shared<TestEncryptionProperties>("a"));
  TestEncrypterFactory factory;
  auto handler = EncryptionHandler::create(type, spec, &factory);
  ASSERT_EQ(handler->getEncryptionGroupCount(), 1);
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_TRUE(handler->isEncrypted(i));
    ASSERT_EQ(handler->getEncryptionRoot(i), 0);
  }
}

TEST(Encryption, invalidField) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  spec.withEncryptedField(
      FieldEncryptionSpecification{}.withEncryptionProperties(
          std::make_shared<TestEncryptionProperties>("a")));
  TestEncrypterFactory factory;
  ASSERT_THROW(
      EncryptionHandler::create(type, spec, &factory),
      exception::LoggedException);
}

TEST(Encryption, invalidProperties) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  spec.withEncryptedField(FieldEncryptionSpecification{}.withIndex(1));
  TestEncrypterFactory factory;
  ASSERT_THROW(
      EncryptionHandler::create(type, spec, &factory),
      exception::LoggedException);
}

TEST(Encryption, differentEncrypter) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int,b:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  spec.withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(0).withEncryptionProperties(
              std::make_shared<TestEncryptionProperties>("a")))
      .withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(1).withEncryptionProperties(
              std::make_shared<TestEncryptionProperties>("b")));
  TestEncrypterFactory factory;
  auto handler = EncryptionHandler::create(type, spec, &factory);
  ASSERT_EQ(handler->getEncryptionGroupCount(), 2);
}

TEST(Encryption, sameEncrypter) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int,b:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  auto props = std::make_shared<TestEncryptionProperties>("a");
  spec.withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(0).withEncryptionProperties(
              props))
      .withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(1).withEncryptionProperties(
              props));
  TestEncrypterFactory factory;
  auto handler = EncryptionHandler::create(type, spec, &factory);
  ASSERT_EQ(handler->getEncryptionGroupCount(), 1);
}

TEST(Encryption, sameEncrypter2) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int,b:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  spec.withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(0).withEncryptionProperties(
              std::make_shared<TestEncryptionProperties>("a")))
      .withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(1).withEncryptionProperties(
              std::make_shared<TestEncryptionProperties>("a")));
  TestEncrypterFactory factory;
  auto handler = EncryptionHandler::create(type, spec, &factory);
  ASSERT_EQ(handler->getEncryptionGroupCount(), 1);
}

TEST(Encryption, dupeField) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int,b:int>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  auto props = std::make_shared<TestEncryptionProperties>("a");
  spec.withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(1).withEncryptionProperties(
              props))
      .withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(1).withEncryptionProperties(
              props));
  TestEncrypterFactory factory;
  ASSERT_THROW(
      EncryptionHandler::create(type, spec, &factory),
      exception::LoggedException);
}

TEST(Encryption, basic) {
  HiveTypeParser parser;
  auto type = parser.parse("struct<a:int,b:float,c:string,d:double>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  for (auto i = 0; i < 4; ++i) {
    if (i % 2 == 1) {
      spec.withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(i).withEncryptionProperties(
              std::make_shared<TestEncryptionProperties>(
                  folly::to<std::string>(i))));
    }
  }
  TestEncrypterFactory factory;
  auto handler = EncryptionHandler::create(type, spec, &factory);
  ASSERT_TRUE(handler->isEncrypted());
  ASSERT_EQ(handler->getKeyProviderType(), EncryptionProvider::Unknown);
  ASSERT_EQ(handler->getEncryptionGroupCount(), 2);
  auto groupIndex = 0;
  for (auto i = 0; i < 4; ++i) {
    ASSERT_EQ(handler->isEncrypted(i + 1), (i % 2 == 1));
    if (i % 2 == 1) {
      ASSERT_EQ(handler->getEncryptionRoot(i + 1), i + 1);
      ASSERT_EQ(handler->getEncryptionGroupIndex(i + 1), groupIndex++);
    }
  }
}

TEST(Encryption, nestedType) {
  HiveTypeParser parser;
  auto type = parser.parse(
      "struct<a:int,b:map<float,map<int,double>>,c:struct<a:string,b:int>,d:array<double>>");
  EncryptionSpecification spec{EncryptionProvider::Unknown};
  for (auto i = 0; i < 4; ++i) {
    if (i % 2 == 1) {
      spec.withEncryptedField(
          FieldEncryptionSpecification{}.withIndex(i).withEncryptionProperties(
              std::make_shared<TestEncryptionProperties>(
                  folly::to<std::string>(i))));
    }
  }

  auto withId = TypeWithId::create(type);
  TestEncrypterFactory factory;
  auto handler = EncryptionHandler::create(type, spec, &factory);
  ASSERT_TRUE(handler->isEncrypted());
  ASSERT_EQ(handler->getKeyProviderType(), EncryptionProvider::Unknown);
  ASSERT_EQ(handler->getEncryptionGroupCount(), 2);
  for (auto i = 0; i < withId->size(); ++i) {
    auto& col = withId->childAt(i);
    bool encrypted = (col->column() % 2 == 1);
    for (auto j = col->id(); j <= col->maxId(); ++j) {
      ASSERT_EQ(handler->isEncrypted(j), encrypted);
      if (encrypted) {
        ASSERT_EQ(handler->getEncryptionRoot(j), col->id());
      }
    }
  }
}
