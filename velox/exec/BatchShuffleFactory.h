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

#include <folly/io/IOBuf.h>
#include <velox/common/memory/Memory.h>
#include <memory>

namespace facebook::velox::exec {

// This class abstracts the shuffle writer interface
class BatchShuffleWriter {
 public:
  /// The writer needs to use a Velox memory pool to allocate or release its
  /// memory
  /// \param pool Velox memory pool
  BatchShuffleWriter(velox::memory::MemoryPool* pool) : pool_(pool) {}

  /// Single row (key, value) write API
  /// \param key the serialized key (aggregated keys needed for sorted shuffle)
  /// \param value the serialized value (row)
  /// \param partition the destination partition
  /// \return true if no error
  virtual bool collect(
      const folly::IOBuf& key,
      const folly::IOBuf& value,
      uint32_t partition) = 0;

  /// Single row (value only) write API
  /// \param value the serialized row value
  /// \param partition the destination partition
  /// \return ture if no error
  virtual bool collect(const folly::IOBuf& value, int partition) = 0;

  /// Reports ends of writing to all partitions to shuffle client
  /// \param success true if everything is good on the caller
  /// \return true on success
  virtual bool close(bool success) = 0;

  virtual ~BatchShuffleWriter() = default;

 protected:
  velox::memory::MemoryPool* pool_;
};

// This class abstracts the shuffle reader iterator interface
class ShuffleReadIterator {
 public:
  /// Returns true if there is more rows
  virtual bool hasNext() = 0;

  /// Advances to the next row (key/value)
  virtual bool next() = 0;

  /// Returns the key of the current row
  virtual std::unique_ptr<folly::IOBuf> getKey() = 0;

  /// Returns the value of the current row
  virtual std::unique_ptr<folly::IOBuf> getValue() = 0;

  /// Closes the iterator after all data is processed
  /// \param success True if all data is read successfully
  /// \return true if the iterator is closed successfully
  /// False indicates that the reader may open the read client
  /// again on the same partition and UoWs
  bool close(bool success);

  virtual ~ShuffleReadIterator() = default;
};

// A handle for shuffle blocks
class BatchBlock {
 public:
  /// Returns the size of this serialized block
  virtual size_t size() const = 0;

  /// Returns the row iterator in this block
  virtual std::unique_ptr<ShuffleReadIterator> iterator() = 0;

  virtual ~BatchBlock() = default;
};

class BatchShuffleBlockIterator {
 public:
  /// Returns true if there is more row block
  virtual bool hasNext() = 0;

  /// Advances to the next row block
  virtual bool next() = 0;

  // Return the block handler
  virtual std::unique_ptr<BatchBlock> block() = 0;

  /// Closes the iterator after all blocks are processed
  /// \param success True if all blocks are read successfully
  /// \return true if the iterator is closed successfully
  /// False indicates that the reader may open the read client
  /// again on the same partition and UoWs
  bool close(bool success);

  virtual ~BatchShuffleBlockIterator() = default;
};

// This class abstracts the shuffle reader interface
class BatchShuffleReader {
 public:
  BatchShuffleReader(velox::memory::MemoryPool* pool) : pool_(pool) {}

  /// Gets an iterator containing the next requested partition and units of work
  /// \param partition Requested partition
  /// \param unitsOfWork vector if UoWs
  /// \return An iterator if successful
  virtual std::unique_ptr<ShuffleReadIterator> getPartition(
      uint32_t partition,
      const std::vector<long> unitsOfWork) = 0;

  /// Gets a block iterator containing the next requested partition and units of
  /// work \param partition Requested partition \param unitsOfWork vector if
  /// UoWs \return An iterator of row blocks if successful
  virtual std::unique_ptr<BatchShuffleBlockIterator> getPartitionBlocks(
      uint32_t partition,
      const std::vector<long> unitsOfWork) = 0;

  /// Close the reader client on completion
  /// \param success True if all data is read successfully
  /// All the iterator data and buffer pointers remain valid
  /// up until this call
  /// \return true if connection closed successfully
  /// Success set to false indicates that the reader may open the read client
  /// again on the same partition and UoWs
  virtual bool close(bool success) = 0;

  virtual ~BatchShuffleReader() = default;

 protected:
  // Velox memory pool
  velox::memory::MemoryPool* pool_;
};

// This class abstracts a shuffle factory for batch shuffle
// By using create functions for both reader and reducer
class BatchShuffleFactory {
 public:
  BatchShuffleFactory() = default;

  /// Factory method to create Shuffle reader
  /// param id an id associated with the reader
  /// The real definition of id is up to the shuffle system
  /// \param pool velox memory pool
  /// \return pointer to the reader
  virtual std::unique_ptr<BatchShuffleReader> createShuffleReader(
      int id,
      std::shared_ptr<velox::memory::MemoryPool> pool) = 0;

  /// Factory method to create Shuffle writer
  /// \param id an id associated with the writer
  /// The real definition of id is up to the shuffle system
  /// \param pool velox memory pool
  /// \return pointer to the writer
  virtual std::unique_ptr<BatchShuffleWriter> createShuffleWriter(
      int id,
      std::shared_ptr<velox::memory::MemoryPool> pool) = 0;

  virtual ~BatchShuffleFactory() = default;
};
} // namespace facebook::velox::exec