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
#include <gtest/gtest.h>
#include "velox/common/base/VeloxException.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook {
namespace {

template <typename T>
void verifyException(
    const T& e,
    const std::string& exceptionName,
    const std::string& fileName,
    const std::string& fileLine,
    const std::string& functionName,
    const std::string& failingExpression,
    const std::string& errorMessage,
    const std::string& errorSource,
    const std::string& errorCode,
    const std::string& retryable,
    const std::string& additionalMessage = "") {
  EXPECT_EQ(fileName, e.fileName());
  if (!fileLine.empty()) {
    EXPECT_EQ(fileLine, folly::to<std::string>(e.fileLine()));
  }
  EXPECT_EQ(functionName, e.functionName());
  EXPECT_EQ(failingExpression, e.failingExpression());
  EXPECT_EQ(errorMessage, e.errorMessage());
  EXPECT_EQ(errorSource, e.errorSource());
  EXPECT_EQ(errorCode, e.errorCode());
  EXPECT_EQ(retryable, e.retryable() ? "True" : "False");

  EXPECT_NE(
      std::string(e.what()).find(exceptionName + "\n"), std::string::npos);
  EXPECT_NE(
      std::string(e.what()).find("Error Source: " + errorSource + "\n"),
      std::string::npos);
  EXPECT_NE(
      std::string(e.what()).find("Error Code: " + errorCode + "\n"),
      std::string::npos);
  if (!errorMessage.empty()) {
    EXPECT_NE(
        std::string(e.what()).find("Error Message: " + errorMessage + "\n"),
        std::string::npos);
  }
  EXPECT_NE(
      std::string(e.what()).find("Retryable: " + retryable + "\n"),
      std::string::npos);
  EXPECT_NE(
      std::string(e.what()).find(
          "Location: " +
          folly::to<std::string>(functionName, '@', fileName, ':', fileLine)),
      std::string::npos);
  if (!failingExpression.empty()) {
    EXPECT_NE(
        std::string(e.what()).find("Expression: " + failingExpression + "\n"),
        std::string::npos);
  }
  EXPECT_NE(std::string(e.what()).find("Stack Trace:\n"), std::string::npos);

  if (!additionalMessage.empty()) {
    EXPECT_NE(std::string(e.what()).find(additionalMessage), std::string::npos);
  }
}

TEST(ExceptionTests, format) {
  verifyException(
      nimble::NimbleUserError(
          "file1", 23, "func1", "expr1", "err1", "code1", true),
      "NimbleUserError",
      "file1",
      "23",
      "func1",
      "expr1",
      "err1",
      "USER",
      "code1",
      "True");

  verifyException(
      nimble::NimbleInternalError(
          "file2", 24, "func2", "expr2", "err2", "code2", false),
      "NimbleInternalError",
      "file2",
      "24",
      "func2",
      "expr2",
      "err2",
      "INTERNAL",
      "code2",
      "False");
}

TEST(ExceptionTests, check) {
  int a = 5;
  try {
    NIMBLE_CHECK(a < 3, "error message1");
  } catch (const nimble::NimbleInternalError& e) {
    verifyException(
        e,
        "NimbleInternalError",
        __FILE__,
        "",
        "TestBody",
        "a < 3",
        "error message1",
        "INTERNAL",
        "INVALID_ARGUMENT",
        "False");
  }
}

TEST(ExceptionTests, unreachable) {
  try {
    NIMBLE_UNREACHABLE("error message");
  } catch (const nimble::NimbleInternalError& e) {
    verifyException(
        e,
        "NimbleInternalError",
        __FILE__,
        "",
        "TestBody",
        "",
        "error message",
        "INTERNAL",
        "UNREACHABLE_CODE",
        "False");
  }
}

TEST(ExceptionTests, notImplemented) {
  try {
    NIMBLE_NOT_IMPLEMENTED("error message7");
  } catch (const nimble::NimbleInternalError& e) {
    verifyException(
        e,
        "NimbleInternalError",
        __FILE__,
        "",
        "TestBody",
        "",
        "error message7",
        "INTERNAL",
        "NOT_IMPLEMENTED",
        "False");
  }
}

TEST(ExceptionTests, notSupported) {
  try {
    NIMBLE_UNSUPPORTED("error message6");
  } catch (const nimble::NimbleUserError& e) {
    verifyException(
        e,
        "NimbleUserError",
        __FILE__,
        "",
        "TestBody",
        "",
        "error message6",
        "USER",
        "NOT_SUPPORTED",
        "False");
  }
}

TEST(ExceptionTests, stackTraceThreads) {
  // Make sure captured stack trace doesn't need anything from thread local
  // storage
  std::exception_ptr e;
  auto throwFunc = []() { NIMBLE_CHECK(false, "Test."); };
  std::thread t([&]() {
    try {
      throwFunc();
    } catch (...) {
      e = std::current_exception();
    }
  });

  t.join();

  ASSERT_NE(nullptr, e);
  EXPECT_NE(
      std::string::npos,
      folly::exceptionStr(e).find(
          "facebook::nimble::NimbleException::NimbleException"));
}

TEST(ExceptionTests, context) {
  auto messageFunc = [](velox::VeloxException::Type exceptionType, void* arg) {
    auto msg = *static_cast<const std::string*>(arg);
    switch (exceptionType) {
      case velox::VeloxException::Type::kUser:
        return fmt::format("USER {}", msg);
      case velox::VeloxException::Type::kSystem:
        return fmt::format("SYSTEM {}", msg);
    }
  };
  std::string context1Message = "1";
  velox::ExceptionContextSetter context1({messageFunc, &context1Message, true});
  std::string context2Message = "2";
  velox::ExceptionContextSetter context2(
      {messageFunc, &context2Message, false});
  std::string context3Message = "3";
  velox::ExceptionContextSetter context3(
      {messageFunc, &context3Message, false});

  try {
    NIMBLE_UNSUPPORTED("");
    FAIL();
  } catch (const nimble::NimbleException& e) {
    ASSERT_EQ(e.context(), "USER 3 USER 1");
    ASSERT_NE(
        std::string(e.what()).find("Context: " + e.context()),
        std::string::npos);
  }

  try {
    NIMBLE_UNKNOWN("");
    FAIL();
  } catch (const nimble::NimbleException& e) {
    ASSERT_EQ(e.context(), "SYSTEM 3 SYSTEM 1");
    ASSERT_NE(
        std::string(e.what()).find("Context: " + e.context()),
        std::string::npos);
  }
}

TEST(ExceptionTests, checkComparisons) {
  // Test NIMBLE_CHECK_GT
  try {
    NIMBLE_CHECK_GT(5, 10);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(std::string(e.what()).find("(5 vs. 10)"), std::string::npos);
    EXPECT_EQ(e.errorCode(), "INVALID_ARGUMENT");
  }

  // Test NIMBLE_CHECK_GE
  try {
    NIMBLE_CHECK_GE(3, 5);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(std::string(e.what()).find("(3 vs. 5)"), std::string::npos);
  }

  // Test NIMBLE_CHECK_LT
  try {
    NIMBLE_CHECK_LT(10, 5);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(std::string(e.what()).find("(10 vs. 5)"), std::string::npos);
  }

  // Test NIMBLE_CHECK_LE
  try {
    NIMBLE_CHECK_LE(8, 3);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(std::string(e.what()).find("(8 vs. 3)"), std::string::npos);
  }

  // Test NIMBLE_CHECK_EQ
  try {
    NIMBLE_CHECK_EQ(5, 10);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(std::string(e.what()).find("(5 vs. 10)"), std::string::npos);
  }

  // Test NIMBLE_CHECK_NE
  try {
    NIMBLE_CHECK_NE(7, 7);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(std::string(e.what()).find("(7 vs. 7)"), std::string::npos);
  }

  // Test successful comparison (should not throw)
  NIMBLE_CHECK_GT(10, 5);
  NIMBLE_CHECK_GE(5, 5);
  NIMBLE_CHECK_LT(3, 8);
  NIMBLE_CHECK_LE(4, 4);
  NIMBLE_CHECK_EQ(7, 7);
  NIMBLE_CHECK_NE(3, 5);
}

TEST(ExceptionTests, checkComparisonsWithCustomMessage) {
  // Test with custom format message
  try {
    NIMBLE_CHECK_GT(5, 10, "custom message: {} items", 42);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    std::string what = e.what();
    EXPECT_NE(what.find("(5 vs. 10)"), std::string::npos);
    EXPECT_NE(what.find("custom message: 42 items"), std::string::npos);
  }

  // Test NIMBLE_CHECK_EQ with custom message
  try {
    NIMBLE_CHECK_EQ(100, 200, "Size mismatch");
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    std::string what = e.what();
    EXPECT_NE(what.find("(100 vs. 200)"), std::string::npos);
    EXPECT_NE(what.find("Size mismatch"), std::string::npos);
  }
}

TEST(ExceptionTests, checkNull) {
  int* nullPtr = nullptr;
  int value = 42;
  int* validPtr = &value;

  // Test NIMBLE_CHECK_NULL - should throw when pointer is not null
  try {
    NIMBLE_CHECK_NULL(validPtr);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_EQ(e.errorCode(), "INVALID_ARGUMENT");
  }

  // Test NIMBLE_CHECK_NULL - should not throw when pointer is null
  NIMBLE_CHECK_NULL(nullPtr);

  // Test NIMBLE_CHECK_NOT_NULL - should throw when pointer is null
  try {
    NIMBLE_CHECK_NOT_NULL(nullPtr);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_EQ(e.errorCode(), "INVALID_ARGUMENT");
  }

  // Test NIMBLE_CHECK_NOT_NULL - should not throw when pointer is valid
  NIMBLE_CHECK_NOT_NULL(validPtr);
}

TEST(ExceptionTests, checkNullWithMessage) {
  int* nullPtr = nullptr;
  int value = 42;
  int* validPtr = &value;

  // Test with custom message
  try {
    NIMBLE_CHECK_NULL(validPtr, "Expected null pointer");
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(
        std::string(e.what()).find("Expected null pointer"), std::string::npos);
  }

  try {
    NIMBLE_CHECK_NOT_NULL(nullPtr, "Pointer should not be null");
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(
        std::string(e.what()).find("Pointer should not be null"),
        std::string::npos);
  }
}

TEST(ExceptionTests, failMacros) {
  // Test NIMBLE_FAIL
  try {
    NIMBLE_FAIL("Internal error: {}", 42);
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(
        std::string(e.what()).find("Internal error: 42"), std::string::npos);
    EXPECT_EQ(e.errorCode(), "INVALID_STATE");
  }

  // Test NIMBLE_USER_FAIL
  try {
    NIMBLE_USER_FAIL("User error: {} is invalid", "input");
    FAIL() << "Should have thrown";
  } catch (const nimble::NimbleUserError& e) {
    EXPECT_NE(
        std::string(e.what()).find("User error: input is invalid"),
        std::string::npos);
    EXPECT_EQ(e.errorCode(), "INVALID_ARGUMENT");
  }
}

} // namespace
} // namespace facebook
