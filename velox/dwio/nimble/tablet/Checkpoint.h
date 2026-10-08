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

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble {

/// Marks a Nimble file as suspended rather than finalized: closed so that a
/// later writer can reopen it for append, instead of sealed.
///
/// A suspended file has a valid footer and remains structurally inspectable.
/// It may omit serving artifacts that require the complete file. A resumed
/// writer rebuilds ordinary state from the sections a reader already consumes
/// and restores only the remaining builder state from this class.
///
/// It is meant to stay that way. The write-path implementation should widen
/// this only after establishing that the existing sections cannot recover the
/// state, or that a new field provides a stronger validation than those
/// sections allow.
///
/// Presence of the `columnar.checkpoint` optional section is the signal that
/// a file is not final. A finalized file never carries it.
class Checkpoint {
 public:
  /// Builder state spilled by an index writer whose index has no incremental
  /// on-disk form. Opaque to the reader: only the index implementation that
  /// produced it can interpret the payload at 'state'.
  struct IndexSpill {
    /// Matches index::IndexFamily. Held as a scalar so the tablet layer does
    /// not depend on the index layer.
    uint8_t family{0};

    /// Index implementation name, matching the name the finalized index
    /// section would carry.
    std::string name;

    /// Version of the spill payload, owned by the index implementation.
    uint32_t stateVersion{0};

    /// Location of the spilled payload.
    MetadataSection state;
  };

  /// Builder state for a file-scope shared dictionary. Only the dictionary
  /// implementation can interpret the payload at 'state'.
  struct DictionarySpill {
    /// Identifier matching the dictionary_id the finalized catalog records.
    uint32_t dictionaryId{0};

    /// Nimble DataType of the alphabet values.
    uint8_t dataType{0};

    /// Version of the spill payload, owned by the dictionary implementation.
    uint32_t stateVersion{0};

    /// Location of the spilled builder state.
    MetadataSection state;
  };

  /// Returns the format version of the checkpoint that was read.
  uint32_t version() const {
    return version_;
  }

  /// Returns the spilled builder state for the indexes this file did not
  /// materialize.
  const std::vector<IndexSpill>& indexSpills() const {
    return indexSpills_;
  }

  /// Returns the next stream offset SchemaBuilder would allocate. Not
  /// derivable from the schema alone; see Checkpoint.fbs.
  uint32_t nextStreamOffset() const {
    return nextStreamOffset_;
  }

  /// Returns the file-scope shared dictionary states captured mid-write.
  const std::vector<DictionarySpill>& dictionarySpills() const {
    return dictionarySpills_;
  }

  /// Deserializes the `columnar.checkpoint` optional section. Throws when the
  /// section fails flatbuffers verification, or when it was written by a
  /// newer format version than this build knows.
  static Checkpoint deserialize(std::string_view data);

 private:
  Checkpoint() = default;

  uint32_t version_{0};
  std::vector<IndexSpill> indexSpills_;
  uint32_t nextStreamOffset_{0};
  std::vector<DictionarySpill> dictionarySpills_;
};

} // namespace facebook::nimble
