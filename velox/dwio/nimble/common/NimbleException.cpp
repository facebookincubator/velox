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

#include "velox/dwio/nimble/common/NimbleException.h"

#include <folly/Conv.h>
#include <folly/debugging/symbolizer/Symbolizer.h>
#include <glog/logging.h>

namespace facebook::nimble {

namespace {

std::string captureContextMessage(
    velox::VeloxException::Type veloxExceptionType) {
  auto* context = &velox::getExceptionContext();
  std::string contextMessage = context->message(veloxExceptionType);
  while (context->parent) {
    context = context->parent;
    if (!context->isEssential) {
      continue;
    }
    const auto message = context->message(veloxExceptionType);
    if (message.empty()) {
      continue;
    }
    if (!contextMessage.empty()) {
      contextMessage += ' ';
    }
    contextMessage += message;
  }
  return contextMessage;
}

} // namespace

NimbleException::NimbleException(
    std::string_view exceptionName,
    const char* fileName,
    size_t fileLine,
    const char* functionName,
    std::string_view failingExpression,
    std::string_view errorMessage,
    std::string_view errorCode,
    bool retryable,
    velox::VeloxException::Type veloxExceptionType)
    : exceptionName_{std::move(exceptionName)},
      fileName_{fileName},
      fileLine_{fileLine},
      functionName_{functionName},
      failingExpression_{std::move(failingExpression)},
      errorMessage_{std::move(errorMessage)},
      errorCode_{std::move(errorCode)},
      retryable_{retryable},
      context_{captureContextMessage(veloxExceptionType)} {
  captureStackTraceFrames();
}

const char* NimbleException::what() const noexcept {
  try {
    folly::call_once(once_, [&] { finalizeMessage(); });
    return finalizedMessage_.c_str();
  } catch (...) {
    return "<unknown failure in NimbleException::what>";
  }
}

void NimbleException::captureStackTraceFrames() {
  try {
    constexpr size_t skipFrames = 2;
    constexpr size_t maxFrames = 200;
    uintptr_t addresses[maxFrames];
    ssize_t n = folly::symbolizer::getStackTrace(addresses, maxFrames);

    if (n < skipFrames) {
      return;
    }

    exceptionFrames_.assign(addresses + skipFrames, addresses + n);
  } catch (const std::exception& ex) {
    LOG(WARNING) << "Unable to capture stack trace: " << ex.what();
  } catch (...) {
    LOG(WARNING) << "Unable to capture stack trace.";
  }
}

void NimbleException::finalizeMessage() const {
  finalizedMessage_ += exceptionName_;
  finalizedMessage_ += "\nError Source: ";
  finalizedMessage_ += errorSource();
  finalizedMessage_ += "\nError Code: ";
  finalizedMessage_ += errorCode_;
  if (!errorMessage_.empty()) {
    finalizedMessage_ += "\nError Message: ";
    finalizedMessage_ += errorMessage_;
  }
  finalizedMessage_ += "\n";
  appendMessage(finalizedMessage_);
  finalizedMessage_ += "Retryable: ";
  finalizedMessage_ += retryable_ ? "True" : "False";
  finalizedMessage_ += "\nLocation: ";
  finalizedMessage_ += functionName_;
  finalizedMessage_ += "@";
  finalizedMessage_ += fileName_;
  finalizedMessage_ += ":";
  finalizedMessage_ += folly::to<std::string>(fileLine_);

  if (!failingExpression_.empty()) {
    finalizedMessage_ += "\nExpression: ";
    finalizedMessage_ += failingExpression_;
  }

  if (!context_.empty()) {
    finalizedMessage_ += "\nContext: ";
    finalizedMessage_ += context_;
  }

  if (LIKELY(!exceptionFrames_.empty())) {
    std::vector<folly::symbolizer::SymbolizedFrame> symbolizedFrames;
    symbolizedFrames.resize(exceptionFrames_.size());

    folly::symbolizer::Symbolizer symbolizer{
        folly::symbolizer::LocationInfoMode::FULL};
    symbolizer.symbolize(
        exceptionFrames_.data(),
        symbolizedFrames.data(),
        symbolizedFrames.size());

    folly::symbolizer::StringSymbolizePrinter printer{
        folly::symbolizer::StringSymbolizePrinter::COLOR};
    printer.println(symbolizedFrames.data(), symbolizedFrames.size());

    finalizedMessage_ += "\nStack Trace:\n";
    finalizedMessage_ += printer.str();
  }
}

NimbleUserError::NimbleUserError(
    const char* fileName,
    size_t fileLine,
    const char* functionName,
    std::string_view failingExpression,
    std::string_view errorMessage,
    std::string_view errorCode,
    bool retryable)
    : NimbleException(
          "NimbleUserError",
          fileName,
          fileLine,
          functionName,
          failingExpression,
          errorMessage,
          errorCode,
          retryable,
          velox::VeloxException::Type::kUser) {}

const std::string_view NimbleUserError::errorSource() const {
  return "USER";
}

NimbleInternalError::NimbleInternalError(
    const char* fileName,
    size_t fileLine,
    const char* functionName,
    std::string_view failingExpression,
    std::string_view errorMessage,
    std::string_view errorCode,
    bool retryable)
    : NimbleException(
          "NimbleInternalError",
          fileName,
          fileLine,
          functionName,
          failingExpression,
          errorMessage,
          errorCode,
          retryable,
          velox::VeloxException::Type::kSystem) {}

const std::string_view NimbleInternalError::errorSource() const {
  return "INTERNAL";
}
} // namespace facebook::nimble
