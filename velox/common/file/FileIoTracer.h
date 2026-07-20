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

#include <fmt/format.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox {

/// Represents a tag in the IO call stack. Tags are linked together to form
/// a stack trace of IO operations, useful for debugging and tracing.
struct IoTag {
  /// The name/label for this tag in the call stack.
  std::string name;

  /// Pointer to the parent tag in the call stack. nullptr if this is the root.
  const IoTag* parent{nullptr};

  IoTag() = default;

  explicit IoTag(std::string_view tagName, const IoTag* parentTag = nullptr)
      : name(tagName), parent(parentTag) {}

  /// Returns the full call stack as a string, with tags separated by " -> ".
  /// Example: "TableScan -> ColumnReader -> PrefixEncoding"
  std::string toString() const;

  /// Returns the depth of this tag in the call stack (1-based).
  size_t depth() const;
};

/// Thread-local IO tag for the current call stack.
/// Use ScopedIoTag to automatically push/pop tags.
const IoTag*& threadIoTag();

/// RAII helper to push/pop IoTag onto the thread-local stack automatically.
/// Usage:
///   ScopedIoTag scopedTag("ColumnReader");
///   // threadIoTag() now points to the new tag
///   // When scopedTag goes out of scope, threadIoTag() is restored
class ScopedIoTag {
 public:
  explicit ScopedIoTag(std::string_view name);

  ~ScopedIoTag();

  ScopedIoTag(const ScopedIoTag&) = delete;
  ScopedIoTag& operator=(const ScopedIoTag&) = delete;

  const IoTag& tag() const;

 private:
  IoTag tag_;
};

/// Type of IO operation.
enum class IoType {
  Read,
  AsyncRead,
  Write,
  AsyncWrite,
};

/// Returns a string representation of the IoType.
std::string toString(IoType type);

/// Stream output operator for IoType.
std::ostream& operator<<(std::ostream& os, IoType type);

/// Abstract tracer for file IO operations. Implementations can record IO
/// operations for debugging, performance analysis, or monitoring purposes.
/// The IO tag is captured from thread-local storage via threadIoTag().
class FileIoTracer {
 public:
  virtual ~FileIoTracer() = default;

  /// Records an IO operation.
  /// @param type The type of IO operation (read, asyncRead, write, asyncWrite).
  /// @param offset The byte offset in the file where the IO starts.
  /// @param length The number of bytes involved in the IO operation.
  /// Note: The IO tag is captured from threadIoTag() by the implementation.
  virtual void record(IoType type, uint64_t offset, uint64_t length) = 0;

  /// Called when all IO operations are complete.
  /// Implementations can use this to flush buffers, finalize reports, etc.
  virtual void finish() = 0;
};

/// Record of a single IO operation captured by InMemoryFileIoTracer.
struct IoRecord {
  IoType type;
  uint64_t offset{};
  uint64_t length{};
  std::string tag;

  /// Returns a string representation of the record.
  /// Format: "<IoType> [<offset>, <length>] <tag>"
  std::string toString() const;
};

/// In-memory file IO tracer that records IO operations to a vector.
/// Useful for testing and debugging. Thread-safe.
class InMemoryFileIoTracer : public FileIoTracer {
 public:
  /// Creates a new InMemoryFileIoTracer with a reference to a vector for
  /// recording.
  /// @param records The vector to record IO operations to.
  static std::shared_ptr<InMemoryFileIoTracer> create(
      std::vector<IoRecord>& records);

  void record(IoType type, uint64_t offset, uint64_t length) override;

  void finish() override;

 private:
  explicit InMemoryFileIoTracer(std::vector<IoRecord>& records);

  std::mutex mutex_;
  std::vector<IoRecord>& records_;
};

} // namespace facebook::velox

/// fmt::formatter specialization for IoType.
template <>
struct fmt::formatter<facebook::velox::IoType> : fmt::formatter<std::string> {
  auto format(facebook::velox::IoType type, fmt::format_context& ctx) const {
    return fmt::formatter<std::string>::format(
        facebook::velox::toString(type), ctx);
  }
};
