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

#include <functional>
#include <memory>

#include "velox/dwio/nimble/writer/BufferPolicy.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble::test {

/// Returns a WriterOptions::bufferPolicyFactory whose policies write the rows
/// of each write() call as one stripe. Tests use it to place stripe boundaries
/// for columns that need a buffer policy, such as Hybrid FlatMap columns.
std::function<std::unique_ptr<BufferPolicy>()>
stripePerWriteBufferPolicyFactory();

std::string createNimbleFile(
    velox::memory::MemoryPool& memoryPool,
    const std::vector<velox::VectorPtr>& vectors,
    nimble::WriterOptions writerOptions = {},
    bool flushAfterWrite = true);

std::string createNimbleFile(
    velox::memory::MemoryPool& memoryPool,
    const velox::VectorPtr& vector,
    nimble::WriterOptions writerOptions = {},
    bool flushAfterWrite = true);

} // namespace facebook::nimble::test
