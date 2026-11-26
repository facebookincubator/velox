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

#include <folly/container/F14Map.h>

#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/nimble/selective/ChunkedDecoder.h"

namespace facebook::nimble {

/// The readers of one key of a FlatMap column: the decoder of the key's in-map
/// stream, null if the stripe has none, and the reader of the key's values.
struct FlatMapKeyNodeBase {
  std::unique_ptr<ChunkedDecoder> inMap;
  std::unique_ptr<velox::dwio::common::SelectiveColumnReader> reader;
};

/// Keeps the key readers of the FlatMap readers in a column reader tree when
/// the tree is destroyed, so that the tree built for another stripe can rebind
/// them to that stripe instead of building new ones. A point lookup reads few
/// rows of each stripe, so building and destroying the readers of the
/// thousands of keys of a stripe can cost more than decoding its rows.
///
/// Keeps the readers of a key until a later stripe has the key again, so its
/// memory grows with the number of distinct keys read. A kept reader holds only
/// its buffers, and none of the data of the stripe it read. Every reader built
/// with a pool must be destroyed before the pool.
///
/// NOTE: this class is not thread-safe. The index reader that owns a pool uses
/// it from one thread at a time.
class FlatMapKeyNodePool {
 public:
  /// Moves the readers kept for the key whose in-map stream is 'inMapStream'
  /// into 'node'. Returns false if none are kept.
  bool take(uint32_t inMapStream, FlatMapKeyNodeBase& node);

  /// Keeps the readers of 'node', the key whose in-map stream is
  /// 'inMapStream', until a later stripe takes them. Unbinds them from the
  /// streams of the stripe they read, so that they keep none of its data
  /// alive. Keeps one set of readers per key.
  void put(uint32_t inMapStream, FlatMapKeyNodeBase& node);

  /// Counts a key node, the readers of a key, that a stripe reused instead of
  /// building.
  void addReusedKeyNodes() {
    ++numReusedKeyNodes_;
  }

  /// Returns the number of key nodes reused instead of built.
  uint64_t numReusedKeyNodes() const {
    return numReusedKeyNodes_;
  }

 private:
  // The kept readers of each key, by the key's in-map stream, which identifies
  // both the key and its FlatMap column in the file.
  folly::F14FastMap<uint32_t, FlatMapKeyNodeBase> nodes_;

  // Key nodes reused instead of built, reported as numReusedFlatMapKeyNodes.
  uint64_t numReusedKeyNodes_{0};
};

} // namespace facebook::nimble
