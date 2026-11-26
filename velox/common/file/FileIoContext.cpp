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

#include "velox/common/file/FileIoContext.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox {
namespace {

using ScopedFileIoContextData = folly::ImmutableRequestData<FileIoContext>;

const folly::RequestToken& scopedContextToken() {
  static const folly::RequestToken kToken{"velox.scoped_file_io_context"};
  return kToken;
}

// Returns the context of the active scope, or nullptr when no scope is active.
const FileIoContext* activeContext() {
  const auto* requestContext = folly::RequestContext::try_get();
  if (requestContext == nullptr) {
    return nullptr;
  }
  const auto* data = static_cast<const ScopedFileIoContextData*>(
      requestContext->getContextData(scopedContextToken()));
  return data == nullptr ? nullptr : &data->value();
}

// Wraps 'context' for the RequestContext. Runs before the scope's guard
// installs anything, so a nested scope throws with the outer one untouched.
std::unique_ptr<folly::RequestData> newScopedFileIoContextData(
    FileIoContext context) {
  VELOX_USER_CHECK_NULL(
      activeContext(), "File IO context scopes must not nest");
  return std::make_unique<ScopedFileIoContextData>(std::move(context));
}

} // namespace

ScopedFileIoContext::ScopedFileIoContext(FileIoContext context)
    : guard_{
          scopedContextToken(),
          newScopedFileIoContextData(std::move(context))} {}

const FileIoContext& ScopedFileIoContext::resolve(
    const FileIoContext& context) {
  const auto* scopedContext = activeContext();
  return scopedContext == nullptr ? context : *scopedContext;
}

} // namespace facebook::velox
