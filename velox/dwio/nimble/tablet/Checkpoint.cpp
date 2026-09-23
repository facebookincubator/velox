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
#include "velox/dwio/nimble/tablet/Checkpoint.h"

#include "flatbuffers/flatbuffers.h"

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/CheckpointGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"

namespace facebook::nimble {

namespace {

MetadataSection toMetadataSection(
    const serialization::MetadataSection* section) {
  const auto uncompressedSize = section->uncompressed_size();
  // No file predates the checkpoint section, so unlike the footer's sections
  // this one never has to tolerate a missing uncompressed size.
  NIMBLE_CHECK_FILE_GT(
      uncompressedSize,
      0,
      "Checkpoint section is missing an uncompressed size.");
  return MetadataSection{
      section->offset(),
      section->size(),
      static_cast<CompressionType>(section->compression_type()),
      uncompressedSize,
  };
}

std::vector<Checkpoint::IndexSpill> toIndexSpills(
    const flatbuffers::Vector<flatbuffers::Offset<serialization::IndexSpill>>*
        spills) {
  std::vector<Checkpoint::IndexSpill> result;
  if (spills == nullptr) {
    return result;
  }
  result.reserve(spills->size());
  for (const auto* spill : *spills) {
    const auto* name = spill->name();
    NIMBLE_CHECK_FILE_NOT_NULL(name, "Index spill is missing its index name.");
    const auto* state = spill->state();
    NIMBLE_CHECK_FILE_NOT_NULL(
        state, "Index spill is missing its state section: {}", name->str());
    result.emplace_back(
        Checkpoint::IndexSpill{
            spill->family(),
            name->str(),
            spill->state_version(),
            toMetadataSection(state),
        });
  }
  return result;
}

std::vector<Checkpoint::DictionarySpill> toDictionarySpills(
    const flatbuffers::Vector<
        flatbuffers::Offset<serialization::DictionarySpill>>* spills) {
  std::vector<Checkpoint::DictionarySpill> result;
  if (spills == nullptr) {
    return result;
  }
  result.reserve(spills->size());
  for (const auto* spill : *spills) {
    const auto* state = spill->state();
    NIMBLE_CHECK_FILE_NOT_NULL(
        state,
        "Dictionary spill is missing its state section: {}",
        spill->dictionary_id());
    result.emplace_back(
        Checkpoint::DictionarySpill{
            spill->dictionary_id(),
            spill->data_type(),
            spill->state_version(),
            toMetadataSection(state),
        });
  }
  return result;
}

} // namespace

Checkpoint Checkpoint::deserialize(std::string_view data) {
  // Unlike the other optional sections, this one is read on a resume path
  // where a torn tail is an expected outcome rather than corruption, so the
  // buffer is verified before it is walked.
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(data.data()), data.size()};
  NIMBLE_CHECK_FILE(
      verifier.VerifyBuffer<serialization::Checkpoint>(nullptr),
      "Corrupt checkpoint section.");

  const auto* serialized = flatbuffers::GetRoot<serialization::Checkpoint>(
      reinterpret_cast<const uint8_t*>(data.data()));

  Checkpoint checkpoint;
  checkpoint.version_ = serialized->checkpoint_version();
  // Widening the table does not belong here: flatbuffers reads a field a
  // writer never wrote as its default, so a reader that gained a field still
  // parses an older checkpoint, and an older reader skips one it does not
  // know. Bump the version only when the meaning of an existing field
  // changes, which this bound would then reject.
  //
  // Below the minimum catches a table whose version field was never written:
  // flatbuffers reads an absent scalar as 0, so a minimal root that passes
  // verification would otherwise deserialize as an empty, plausible-looking
  // checkpoint. Above the maximum catches a newer writer whose state this
  // build cannot interpret, where resuming from a partial understanding of it
  // would corrupt the file.
  NIMBLE_CHECK_FILE(
      checkpoint.version_ >= kCheckpointVersionMin &&
          checkpoint.version_ <= kCheckpointVersion,
      "Unsupported checkpoint version: {}. Supported: {} to {}.",
      checkpoint.version_,
      kCheckpointVersionMin,
      kCheckpointVersion);

  checkpoint.indexSpills_ = toIndexSpills(serialized->index_spills());
  checkpoint.nextStreamOffset_ = serialized->next_stream_offset();
  NIMBLE_CHECK_FILE_GT(
      checkpoint.nextStreamOffset_,
      0,
      "Checkpoint is missing the next stream offset.");

  checkpoint.dictionarySpills_ =
      toDictionarySpills(serialized->dictionary_spills());
  return checkpoint;
}

} // namespace facebook::nimble
