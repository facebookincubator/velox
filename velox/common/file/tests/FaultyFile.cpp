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

#include "velox/common/file/tests/FaultyFile.h"

namespace facebook::velox::tests::utils {

FaultyReadFile::FaultyReadFile(
    const std::string& path,
    std::shared_ptr<ReadFile> delegatedFile,
    FileFaultInjectionHook injectionHook)
    : path_(path),
      delegatedFile_(std::move(delegatedFile)),
      injectionHook_(std::move(injectionHook)) {
  VELOX_CHECK_NOT_NULL(delegatedFile_);
}

std::string_view
FaultyReadFile::pread(uint64_t offset, uint64_t length, void* buf) const {
  if (injectionHook_ != nullptr) {
    FaultFileReadOperation op(path_, offset, length, buf);
    injectionHook_(&op);
    if (!op.delegate) {
      return std::string_view(static_cast<char*>(op.buf), op.length);
    }
  }
  return delegatedFile_->pread(offset, length, buf);
}

uint64_t FaultyReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) const {
  if (injectionHook_ != nullptr) {
    FaultFileReadvOperation op(path_, offset, buffers);
    injectionHook_(&op);
    if (!op.delegate) {
      return op.readBytes;
    }
  }
  return delegatedFile_->preadv(offset, buffers);
}

FaultyWriteFile::FaultyWriteFile(
    std::shared_ptr<WriteFile> delegatedFile,
    FileFaultInjectionHook injectionHook)
    : delegatedFile_(std::move(delegatedFile)),
      injectionHook_(std::move(injectionHook)) {
  VELOX_CHECK_NOT_NULL(delegatedFile_);
}

void FaultyWriteFile::append(std::string_view data) {
  delegatedFile_->append(data);
}

void FaultyWriteFile::append(std::unique_ptr<folly::IOBuf> data) {
  delegatedFile_->append(std::move(data));
}

void FaultyWriteFile::flush() {
  delegatedFile_->flush();
}

void FaultyWriteFile::close() {
  delegatedFile_->close();
}
} // namespace facebook::velox::tests::utils
