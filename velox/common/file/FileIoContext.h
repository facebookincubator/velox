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

#include <memory>
#include <string>
#include <utility>

#include <folly/CancellationToken.h>
#include <folly/container/F14Map.h>
#include <folly/io/async/Request.h>

#include "velox/common/file/FileIoTracer.h"

namespace facebook::velox {

namespace memory {
class MemoryPool;
} // namespace memory

class IoStats;

/// Describes one read to the storage layer. Every ReadFile read (pread,
/// preadv, preadvAsync) takes one by reference and uses it only until the call
/// returns; an asynchronous read copies the fields it needs. 'ioStats' and
/// 'pool' are borrowed: they must outlive the read, which for preadvAsync
/// lasts until its future completes. Set only the fields the read needs:
///
///   file.pread(offset, length, buffer, FileIoContext{.ioStats = &ioStats});
struct FileIoContext {
  /// Stats for IO operations.
  IoStats* ioStats{nullptr};

  /// Options for file read operations.
  folly::F14FastMap<std::string, std::string> fileOpts{};

  /// Tracer for IO operations, providing call stack context.
  std::shared_ptr<FileIoTracer> ioTracer{};

  /// When false, hints to the storage layer that this read should not be cached
  /// or should be evicted soon after reading. This is useful for one-time reads
  /// where caching would waste resources.
  bool cacheable{false};

  /// Pool for scratch memory a read needs only until it returns, such as a
  /// staging buffer. Supplied per call because one ReadFile can serve many
  /// queries, and the memory belongs to the query doing the read. Null means
  /// the file allocates outside any pool.
  memory::MemoryPool* pool{nullptr};

  /// Cancels this read once the query it belongs to is canceled. See ReadFile
  /// for what a canceled read does.
  folly::CancellationToken cancellationToken{};
};

/// Replaces the FileIoContext of reads that honor it: reads on this thread
/// while the scope lives, and work started in it that carries
/// folly::RequestContext, even after it ends. That covers folly thread pools,
/// EventBase, futures, fibers and coroutines started or awaited in the scope,
/// but not ManualExecutor, ThreadedExecutor or a raw std::thread. Nimble's
/// DataInput and MetadataInput honor it; other readers ignore it. Scopes do not
/// nest. Destroy it on the thread, or in the coroutine, that created it.
class ScopedFileIoContext {
 public:
  /// Throws if a scope is already active, including one that work inherited
  /// from an earlier scope. 'context' replaces the reader's whole context, so
  /// it must carry every field the reads need. Its ioStats and pool must
  /// outlive every read started in the scope.
  explicit ScopedFileIoContext(FileIoContext context);

  /// Returns the active scope's context, or 'context' when no scope is active.
  /// The result stays valid until the caller returns; do not store it.
  static const FileIoContext& resolve(const FileIoContext& context);

 private:
  // Installs a shallow copy of the current RequestContext that holds the
  // context, and restores the previous one on destruction. It cannot be copied
  // or moved, so neither can the scope.
  folly::ShallowCopyRequestContextScopeGuard guard_;
};

} // namespace facebook::velox
