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

namespace facebook::velox::detail {

#ifdef _MSC_VER
// Provide complete function specializations for MSVC compatibility

// VeloxRuntimeError specializations
template <>
void veloxCheckFail<::facebook::velox::VeloxRuntimeError, CompileTimeEmptyString>(
    const VeloxCheckFailArgs& args, CompileTimeEmptyString s) {
  LOG(ERROR) << "Line: " << args.file << ":" << args.line
             << ", Function:" << args.function
             << ", Expression: " << args.expression << " " << s
             << ", Source: " << args.errorSource
             << ", ErrorCode: " << args.errorCode;
  ++threadNumVeloxThrow();
  throw ::facebook::velox::VeloxRuntimeError(
      args.file,
      args.line,
      args.function,
      args.expression,
      s,
      args.errorSource,
      args.errorCode,
      args.isRetriable);
}

template <>
void veloxCheckFail<::facebook::velox::VeloxRuntimeError, const char*>(
    const VeloxCheckFailArgs& args, const char* s) {
  LOG(ERROR) << "Line: " << args.file << ":" << args.line
             << ", Function:" << args.function
             << ", Expression: " << args.expression << " " << s
             << ", Source: " << args.errorSource
             << ", ErrorCode: " << args.errorCode;
  ++threadNumVeloxThrow();
  throw ::facebook::velox::VeloxRuntimeError(
      args.file,
      args.line,
      args.function,
      args.expression,
      s,
      args.errorSource,
      args.errorCode,
      args.isRetriable);
}

template <>
void veloxCheckFail<::facebook::velox::VeloxRuntimeError, const std::string&>(
    const VeloxCheckFailArgs& args, const std::string& s) {
  LOG(ERROR) << "Line: " << args.file << ":" << args.line
             << ", Function:" << args.function
             << ", Expression: " << args.expression << " " << s
             << ", Source: " << args.errorSource
             << ", ErrorCode: " << args.errorCode;
  ++threadNumVeloxThrow();
  throw ::facebook::velox::VeloxRuntimeError(
      args.file,
      args.line,
      args.function,
      args.expression,
      s,
      args.errorSource,
      args.errorCode,
      args.isRetriable);
}

// VeloxUserError specializations
template <>
void veloxCheckFail<::facebook::velox::VeloxUserError, CompileTimeEmptyString>(
    const VeloxCheckFailArgs& args, CompileTimeEmptyString s) {
  // No LOG(ERROR) for user errors
  ++threadNumVeloxThrow();
  throw ::facebook::velox::VeloxUserError(
      args.file,
      args.line,
      args.function,
      args.expression,
      s,
      args.errorSource,
      args.errorCode,
      args.isRetriable);
}

template <>
void veloxCheckFail<::facebook::velox::VeloxUserError, const char*>(
    const VeloxCheckFailArgs& args, const char* s) {
  // No LOG(ERROR) for user errors
  ++threadNumVeloxThrow();
  throw ::facebook::velox::VeloxUserError(
      args.file,
      args.line,
      args.function,
      args.expression,
      s,
      args.errorSource,
      args.errorCode,
      args.isRetriable);
}

template <>
void veloxCheckFail<::facebook::velox::VeloxUserError, const std::string&>(
    const VeloxCheckFailArgs& args, const std::string& s) {
  // No LOG(ERROR) for user errors
  ++threadNumVeloxThrow();
  throw ::facebook::velox::VeloxUserError(
      args.file,
      args.line,
      args.function,
      args.expression,
      s,
      args.errorSource,
      args.errorCode,
      args.isRetriable);
}

#else // _MSC_VER

DEFINE_CHECK_FAIL_TEMPLATES(::facebook::velox::VeloxRuntimeError);
DEFINE_CHECK_FAIL_TEMPLATES(::facebook::velox::VeloxUserError);

#endif // _MSC_VER

} // namespace facebook::velox::detail
