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

#include <exception>
#include <string>

#include <folly/Exception.h>
#include <folly/FixedString.h>
#include <folly/String.h>
#include <folly/synchronization/CallOnce.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "velox/common/process/StackTrace.h"

DECLARE_bool(velox_exception_user_stacktrace_enabled);
DECLARE_bool(velox_exception_system_stacktrace_enabled);

DECLARE_int32(velox_exception_user_stacktrace_rate_limit_ms);
DECLARE_int32(velox_exception_system_stacktrace_rate_limit_ms);

namespace facebook {
namespace velox {

namespace error_source {
using namespace folly::string_literals;

// Errors where the root cause of the problem is either because of bad input
// or an unsupported pattern of use are classified with source USER. Examples
// of errors in this category include syntax errors, unavailable names or
// objects.
inline constexpr auto kErrorSourceUser = "USER"_fs;

// Errors where the root cause of the problem is an unexpected internal state in
// the system.
inline constexpr auto kErrorSourceRuntime = "RUNTIME"_fs;

// Errors where the root cause of the problem is some unreliable aspect of the
// system are classified with source SYSTEM.
inline constexpr auto kErrorSourceSystem = "SYSTEM"_fs;
} // namespace error_source

namespace error_code {
using namespace folly::string_literals;

//====================== User Error Codes ======================:

// A generic user error code
inline constexpr auto kGenericUserError = "GENERIC_USER_ERROR"_fs;

// An error raised when an argument verification fails
inline constexpr auto kInvalidArgument = "INVALID_ARGUMENT"_fs;

// An error raised when a requested operation is not supported.
inline constexpr auto kUnsupported = "UNSUPPORTED"_fs;

// Arithmetic errors - underflow, overflow, divide by zero etc.
inline constexpr auto kArithmeticError = "ARITHMETIC_ERROR"_fs;

// Arithmetic errors - underflow, overflow, divide by zero etc.
inline constexpr auto kSchemaMismatch = "SCHEMA_MISMATCH"_fs;

//====================== Runtime Error Codes ======================:

// An error raised when the current state of a component is invalid.
inline constexpr auto kInvalidState = "INVALID_STATE"_fs;

// An error raised when unreachable code point was executed.
inline constexpr auto kUnreachableCode = "UNREACHABLE_CODE"_fs;

// An error raised when a requested operation is not yet supported.
inline constexpr auto kNotImplemented = "NOT_IMPLEMENTED"_fs;

// An error raised when memory exceeded limits.
inline constexpr auto kMemCapExceeded = "MEM_CAP_EXCEEDED"_fs;

// Error caused by memory allocation failure.
inline constexpr auto kMemAllocError = "MEM_ALLOC_ERROR"_fs;

// Error caused by failing to allocate cache buffer space for IO.
inline constexpr auto kNoCacheSpace = "NO_CACHE_SPACE"_fs;

// Errors indicating file read corruptions.
inline constexpr auto kFileCorruption = "FILE_CORRUPTION"_fs;

// We do not know how to classify it yet.
inline constexpr auto kUnknown = "UNKNOWN"_fs;
} // namespace error_code

class VeloxException : public std::exception {
 public:
  enum class Type { kUser = 0, kSystem = 1 };

  VeloxException(
      const char* file,
      size_t line,
      const char* function,
      std::string_view expression,
      std::string_view message,
      std::string_view errorSource,
      std::string_view errorCode,
      bool isRetriable,
      Type exceptionType = Type::kSystem,
      std::string_view exceptionName = "VeloxException");

  /// Wrap an std::exception.
  VeloxException(
      const std::exception_ptr& e,
      std::string_view message,
      std::string_view errorSource,
      bool isRetriable,
      Type exceptionType = Type::kSystem,
      std::string_view exceptionName = "VeloxException");

  // Inherited
  const char* what() const noexcept override {
    return state_->what();
  }

  // Introduced nonvirtuals
  const process::StackTrace* stackTrace() const {
    return state_->stackTrace.get();
  }
  const char* file() const {
    return state_->file;
  }
  size_t line() const {
    return state_->line;
  }
  const char* function() const {
    return state_->function;
  }
  const std::string& failingExpression() const {
    return state_->failingExpression;
  }
  const std::string& message() const {
    return state_->message;
  }

  const std::string& errorCode() const {
    return state_->errorCode;
  }

  const std::string& errorSource() const {
    return state_->errorSource;
  }

  Type exceptionType() const {
    return state_->exceptionType;
  }

  const std::string& exceptionName() const {
    return state_->exceptionName;
  }

  bool isRetriable() const {
    return state_->isRetriable;
  }

  bool isUserError() const {
    return state_->errorSource == error_source::kErrorSourceUser;
  }

  const std::string& context() const {
    return state_->context;
  }

  const std::string& topLevelContext() const {
    return state_->topLevelContext;
  }

  const std::exception_ptr& wrappedException() const {
    return state_->wrappedException;
  }

 private:
  struct State {
    std::unique_ptr<process::StackTrace> stackTrace;
    Type exceptionType = Type::kSystem;
    std::string exceptionName;
    const char* file = nullptr;
    size_t line = 0;
    const char* function = nullptr;
    std::string failingExpression;
    std::string message;
    std::string errorSource;
    std::string errorCode;
    // The current exception context.
    std::string context;
    // The top-level ancestor of the current exception context.
    std::string topLevelContext;
    bool isRetriable;
    // The original std::exception.
    std::exception_ptr wrappedException;

    mutable folly::once_flag once;
    mutable std::string elaborateMessage;

    template <typename F>
    static std::shared_ptr<const State> make(Type exceptionType, F);

    template <typename F>
    static std::shared_ptr<const State> make(F f) {
      auto state = std::make_shared<VeloxException::State>();
      f(*state);
      return state;
    }

    void finalize() const;

    const char* what() const noexcept;
  };

  explicit VeloxException(std::shared_ptr<State const> state) noexcept
      : state_(std::move(state)) {}

  const std::shared_ptr<const State> state_;
};

class VeloxUserError : public VeloxException {
 public:
  VeloxUserError(
      const char* file,
      size_t line,
      const char* function,
      std::string_view expression,
      std::string_view message,
      std::string_view /* errorSource */,
      std::string_view errorCode,
      bool isRetriable,
      std::string_view exceptionName = "VeloxUserError")
      : VeloxException(
            file,
            line,
            function,
            expression,
            message,
            error_source::kErrorSourceUser,
            errorCode,
            isRetriable,
            Type::kUser,
            exceptionName) {}

  /// Wrap an std::exception.
  VeloxUserError(
      const std::exception_ptr& e,
      std::string_view message,
      bool isRetriable,
      std::string_view exceptionName = "VeloxUserError")
      : VeloxException(
            e,
            message,
            error_source::kErrorSourceUser,
            isRetriable,
            Type::kUser,
            exceptionName) {}
};

class VeloxRuntimeError final : public VeloxException {
 public:
  VeloxRuntimeError(
      const char* file,
      size_t line,
      const char* function,
      std::string_view expression,
      std::string_view message,
      std::string_view /* errorSource */,
      std::string_view errorCode,
      bool isRetriable,
      std::string_view exceptionName = "VeloxRuntimeError")
      : VeloxException(
            file,
            line,
            function,
            expression,
            message,
            error_source::kErrorSourceRuntime,
            errorCode,
            isRetriable,
            Type::kSystem,
            exceptionName) {}

  /// Wrap an std::exception.
  VeloxRuntimeError(
      const std::exception_ptr& e,
      std::string_view message,
      bool isRetriable,
      std::string_view exceptionName = "VeloxRuntimeError")
      : VeloxException(
            e,
            message,
            error_source::kErrorSourceRuntime,
            isRetriable,
            Type::kSystem,
            exceptionName) {}
};

/// Holds a pointer to a function that provides addition context to be
/// added to the detailed error message in case of an exception.
struct ExceptionContext {
  using MessageFunction =
      std::string (*)(VeloxException::Type exceptionType, void* arg);

  /// Function to call in case of an exception to get additional context.
  MessageFunction messageFunc{nullptr};

  /// Value to pass to `messageFunc`. Can be null.
  void* arg{nullptr};

  /// Pointer to the parent context when there are hierarchical exception
  /// contexts.
  ExceptionContext* parent{nullptr};

  /// Calls `messageFunc(arg)` and returns the result. Returns empty string if
  /// `messageFunc` is null.
  std::string message(VeloxException::Type exceptionType) {
    if (!messageFunc || suspended) {
      return "";
    }

    std::string theMessage;

    try {
      // Make sure not to call messageFunc again in case it throws.
      suspended = true;
      theMessage = messageFunc(exceptionType, arg);
      suspended = false;
    } catch (...) {
      return "Failed to produce additional context.";
    }

    return theMessage;
  }

  bool suspended{false};
};

/// Returns a reference to thread_local variable that holds a function that can
/// be used to get addition context to be added to the detailed error message in
/// case an exception occurs. This is to used in cases when stack trace would
/// not provide enough information, e.g. in case of hierarchical processing like
/// expression evaluation.
ExceptionContext& getExceptionContext();

/// RAII class to set and restore context for exceptions. Links the new
/// exception context with the previous context held by the thread_local
/// variable to allow retrieving the top-level context when there is an
/// exception context hierarchy.
class ExceptionContextSetter {
 public:
  explicit ExceptionContextSetter(ExceptionContext value)
      : prev_{getExceptionContext()} {
    value.parent = &prev_;
    getExceptionContext() = std::move(value);
  }

  ~ExceptionContextSetter() {
    getExceptionContext() = std::move(prev_);
  }

 private:
  ExceptionContext prev_;
};
} // namespace velox
} // namespace facebook
