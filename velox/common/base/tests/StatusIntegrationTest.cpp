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

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Status.h"

#include <gtest/gtest.h>

namespace facebook::velox::test {
namespace {

Status returnIf(bool cond) {
  VELOX_RETURN_IF(cond, Status::Invalid("error"));
  return Status::OK();
}

Status returnNotOk(Status s) {
  VELOX_RETURN_NOT_OK(s);
  return Status::Invalid("invalid");
}

#define STATUS_MACRO_TEST(name, macro) \
  Status returnMacro##name() {         \
    macro;                             \
    return Status::OK();               \
  }

STATUS_MACRO_TEST(EmptyMessage, VELOX_USER_RETURN_GT(2, 1));
STATUS_MACRO_TEST(Format, VELOX_USER_RETURN_GT(2, 1, "Occurred {} times.", 5));
STATUS_MACRO_TEST(GT, VELOX_USER_RETURN_GT(2, 1, "User error occurred."));
STATUS_MACRO_TEST(GE, VELOX_USER_RETURN_GE(2, 1, "User error occurred."));
STATUS_MACRO_TEST(LT, VELOX_USER_RETURN_LT(1, 2, "User error occurred."));
STATUS_MACRO_TEST(LE, VELOX_USER_RETURN_LE(1, 2, "User error occurred."));
STATUS_MACRO_TEST(EQ, VELOX_USER_RETURN_EQ(1, 1, "User error occurred."));
STATUS_MACRO_TEST(NE, VELOX_USER_RETURN_NE(1, 3, "User error occurred."));
STATUS_MACRO_TEST(
    NULL,
    VELOX_USER_RETURN_NULL(nullptr, "User error occurred."));

Status returnNotNull(Status* status) {
  VELOX_USER_RETURN_NOT_NULL(status, "User error occurred.");
  return Status::OK();
}

Status returnMacroCheck() {
  Status status = Status::OK();
  VELOX_USER_RETURN(
      status.code() != StatusCode::kCancelled, "User error occurred.");
  return Status::OK();
}

TEST(StatusTest, statusMacros) {
  ASSERT_EQ(returnIf(true), Status::Invalid("error"));
  ASSERT_EQ(returnIf(false), Status::OK());

  ASSERT_EQ(returnNotOk(Status::UserError("user")), Status::UserError("user"));
  ASSERT_EQ(returnNotOk(Status::OK()), Status::Invalid("invalid"));

  VELOX_CHECK_OK(Status::OK()); // does not throw.

  bool didThrow = false;
  try {
    VELOX_CHECK_OK(Status::Invalid("invalid"));
  } catch (const VeloxRuntimeError&) {
    didThrow = true;
  }
  ASSERT_TRUE(didThrow) << "VELOX_CHECK_OK did not throw";

  ASSERT_EQ(
      returnMacroCheck(),
      Status::UserError(
          "Reason: User error occurred.\nExpression: status.code() != StatusCode::kCancelled\n"));
  ASSERT_EQ(
      returnMacroEmptyMessage(),
      Status::UserError("Reason: (2 vs. 1)\nExpression: 2 > 1\n"));
  ASSERT_EQ(
      returnMacroFormat(),
      Status::UserError(
          "Reason: (2 vs. 1) Occurred 5 times.\nExpression: 2 > 1\n"));
  ASSERT_EQ(
      returnMacroGT(),
      Status::UserError(
          "Reason: (2 vs. 1) User error occurred.\nExpression: 2 > 1\n"));
  ASSERT_EQ(
      returnMacroGE(),
      Status::UserError(
          "Reason: (2 vs. 1) User error occurred.\nExpression: 2 >= 1\n"));
  ASSERT_EQ(
      returnMacroLT(),
      Status::UserError(
          "Reason: (1 vs. 2) User error occurred.\nExpression: 1 < 2\n"));
  ASSERT_EQ(
      returnMacroLE(),
      Status::UserError(
          "Reason: (1 vs. 2) User error occurred.\nExpression: 1 <= 2\n"));
  ASSERT_EQ(
      returnMacroEQ(),
      Status::UserError(
          "Reason: (1 vs. 1) User error occurred.\nExpression: 1 == 1\n"));
  ASSERT_EQ(
      returnMacroNE(),
      Status::UserError(
          "Reason: (1 vs. 3) User error occurred.\nExpression: 1 != 3\n"));
  ASSERT_EQ(
      returnMacroNULL(),
      Status::UserError(
          "Reason: User error occurred.\nExpression: nullptr == nullptr\n"));
  Status status = Status::OK();
  ASSERT_EQ(
      returnNotNull(&status),
      Status::UserError(
          "Reason: User error occurred.\nExpression: status != nullptr\n"));
}

TEST(StatusTest, statusMacrosSkipDetails) {
  ScopedThreadSkipErrorDetails skipErrorDetails(true);
  ASSERT_EQ(returnMacroCheck(), Status::UserError());
  ASSERT_EQ(returnMacroEmptyMessage(), Status::UserError());
  ASSERT_EQ(returnMacroFormat(), Status::UserError());
  ASSERT_EQ(returnMacroGT(), Status::UserError());
  ASSERT_EQ(returnMacroGE(), Status::UserError());
  ASSERT_EQ(returnMacroLT(), Status::UserError());
  ASSERT_EQ(returnMacroLE(), Status::UserError());
  ASSERT_EQ(returnMacroEQ(), Status::UserError());
  ASSERT_EQ(returnMacroNE(), Status::UserError());
  ASSERT_EQ(returnMacroNULL(), Status::UserError());
  Status status = Status::OK();
  ASSERT_EQ(returnNotNull(&status), Status::UserError());
}

} // namespace
} // namespace facebook::velox::test
