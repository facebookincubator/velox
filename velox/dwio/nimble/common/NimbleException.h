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
#pragma once

#include <folly/FixedString.h>
#include <folly/synchronization/CallOnce.h>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/base/VeloxException.h"

namespace facebook::nimble {

namespace error_source {
using namespace folly::string_literals;

// Errors where the root cause of the problem is either because of bad input
// or an unsupported pattern of use are classified with source USER.
inline constexpr auto User = "USER"_fs;

// Errors where the root cause of the problem is an unexpected internal state in
// the system.
inline constexpr auto Internal = "INTERNAL"_fs;

// Errors where the root cause of the problem is the result of a dependency or
// an environment failures.
inline constexpr auto External = "EXTERNAL"_fs;
} // namespace error_source

namespace error_code {
using namespace folly::string_literals;

// An error raised when an argument verification fails
inline constexpr auto InvalidArgument = "INVALID_ARGUMENT"_fs;

// An error raised when the current state of a component is invalid.
inline constexpr auto InvalidState = "INVALID_STATE"_fs;

// An error raised when unreachable code point was executed.
inline constexpr auto UnreachableCode = "UNREACHABLE_CODE"_fs;

// An error raised when a requested operation is not yet implemented.
inline constexpr auto NotImplemented = "NOT_IMPLEMENTED"_fs;

// An error raised when a requested operation is not supported.
inline constexpr auto NotSupported = "NOT_SUPPORTED"_fs;

// As error raised during encoding optimization, when incompatible encoding is
// attempted.
inline constexpr auto IncompatibleEncoding = "INCOMPATIBLE_ENCODING"_fs;

// An error raised if a corrupted file is detected
inline constexpr auto CorruptedFile = "CORRUPTED_FILE"_fs;

// We do not know how to classify it yet.
inline constexpr auto Unknown = "UNKNOWN"_fs;
} // namespace error_code

namespace external_source {
using namespace folly::string_literals;

// Warm Storage
inline constexpr auto WarmStorage = "WARM_STORAGE"_fs;

// Local File System
inline constexpr auto LocalFileSystem = "FILE_SYSTEM"_fs;

} // namespace external_source

// Base exception used by all other Nimble exceptions. Provides common
// functionality for all other exception types.
class NimbleException : public std::exception {
 public:
  explicit NimbleException(
      std::string_view exceptionName,
      const char* fileName,
      size_t fileLine,
      const char* functionName,
      std::string_view failingExpression,
      std::string_view errorMessage,
      std::string_view errorCode,
      bool retryable,
      velox::VeloxException::Type veloxExceptionType);

  const char* what() const noexcept override;

  const std::string& exceptionName() const {
    return exceptionName_;
  }

  const char* fileName() const {
    return fileName_;
  }

  size_t fileLine() const {
    return fileLine_;
  }

  const char* functionName() const {
    return functionName_;
  }

  const std::string& failingExpression() const {
    return failingExpression_;
  }

  const std::string& errorMessage() const {
    return errorMessage_;
  }

  virtual const std::string_view errorSource() const = 0;

  const std::string& errorCode() const {
    return errorCode_;
  }

  bool retryable() const {
    return retryable_;
  }

  const std::string& context() const {
    return context_;
  }

 protected:
  virtual void appendMessage(std::string& /* message */) const {}

 private:
  void captureStackTraceFrames();
  void finalizeMessage() const;

  std::vector<uintptr_t> exceptionFrames_;
  const std::string stackTrace_;
  const std::string exceptionName_;
  const char* fileName_;
  const size_t fileLine_;
  const char* functionName_;
  const std::string failingExpression_;
  const std::string errorMessage_;
  const std::string errorCode_;
  const bool retryable_;
  const std::string context_;

  mutable folly::once_flag once_;
  mutable std::string finalizedMessage_;
};

// Exception representing an error originating by a user misusing Nimble.
class NimbleUserError : public NimbleException {
 public:
  NimbleUserError(
      const char* fileName,
      size_t fileLine,
      const char* functionName,
      std::string_view failingExpression,
      std::string_view errorMessage,
      std::string_view errorCode,
      bool retryable);

  const std::string_view errorSource() const override;
};

// Exception representing an internal error within the Nimble library.
class NimbleInternalError : public NimbleException {
 public:
  NimbleInternalError(
      const char* fileName,
      size_t fileLine,
      const char* functionName,
      std::string_view failingExpression,
      std::string_view errorMessage,
      std::string_view errorCode,
      bool retryable);

  const std::string_view errorSource() const override;
};
} // namespace facebook::nimble
