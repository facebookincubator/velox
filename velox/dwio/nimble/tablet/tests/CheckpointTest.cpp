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

#include <gtest/gtest.h>
#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/tablet/CheckpointGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"

#include "flatbuffers/flatbuffers.h"

#include <memory>
#include <string>
#include <vector>

namespace facebook::nimble {
namespace {

// Describes one index spill to serialize. Mirrors Checkpoint::IndexSpill, but
// kept separate so a test can build a spill the production type cannot
// represent, such as one whose state section omits its uncompressed size.
struct SpillSpec {
  uint8_t family{0};
  std::string name;
  uint32_t stateVersion{0};
  uint64_t offset{0};
  uint32_t size{0};
  uint32_t uncompressedSize{0};
};

struct DictionarySpillSpec {
  uint32_t dictionaryId{0};
  uint8_t dataType{0};
  uint32_t stateVersion{0};
};

// The residual writer state a Checkpoint carries beyond its version and
// index spills.
struct StateSpec {
  uint32_t nextStreamOffset{1};
  std::vector<DictionarySpillSpec> dictionarySpills;
};

// Serializes a Checkpoint table. There is no production serializer yet, so
// the tests drive the generated builder directly.
std::string serializeCheckpoint(
    uint32_t version,
    const std::vector<SpillSpec>& spills = {},
    const StateSpec& state = {}) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<serialization::IndexSpill>> spillOffsets;
  spillOffsets.reserve(spills.size());
  for (const auto& spill : spills) {
    const auto name = builder.CreateString(spill.name);
    const auto section = serialization::CreateMetadataSection(
        builder,
        spill.offset,
        spill.size,
        static_cast<serialization::CompressionType>(CompressionType::Zstd),
        spill.uncompressedSize);
    spillOffsets.push_back(
        serialization::CreateIndexSpill(
            builder, spill.family, name, spill.stateVersion, section));
  }

  std::vector<flatbuffers::Offset<serialization::DictionarySpill>>
      dictionaryOffsets;
  dictionaryOffsets.reserve(state.dictionarySpills.size());
  for (const auto& spill : state.dictionarySpills) {
    const auto section = serialization::CreateMetadataSection(
        builder,
        /*offset=*/256,
        /*size=*/64,
        static_cast<serialization::CompressionType>(CompressionType::Zstd),
        /*uncompressed_size=*/256);
    dictionaryOffsets.push_back(
        serialization::CreateDictionarySpill(
            builder,
            spill.dictionaryId,
            spill.dataType,
            spill.stateVersion,
            section));
  }

  builder.Finish(
      serialization::CreateCheckpoint(
          builder,
          version,
          builder.CreateVector(spillOffsets),
          state.nextStreamOffset,
          builder.CreateVector(dictionaryOffsets)));
  return std::string{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

TEST(CheckpointTest, basic) {
  const auto checkpoint =
      Checkpoint::deserialize(serializeCheckpoint(kCheckpointVersion));
  EXPECT_EQ(checkpoint.version(), kCheckpointVersion);
  EXPECT_TRUE(checkpoint.indexSpills().empty());
  EXPECT_EQ(checkpoint.nextStreamOffset(), 1);
  EXPECT_TRUE(checkpoint.dictionarySpills().empty());
}

TEST(CheckpointTest, roundTripWriterState) {
  const StateSpec state{
      .nextStreamOffset = 97,
      .dictionarySpills = {
          {.dictionaryId = 3, .dataType = 1, .stateVersion = 4},
          {.dictionaryId = 9, .dataType = 2, .stateVersion = 7},
      }};

  const auto checkpoint = Checkpoint::deserialize(
      serializeCheckpoint(kCheckpointVersion, /*spills=*/{}, state));
  EXPECT_EQ(checkpoint.nextStreamOffset(), state.nextStreamOffset);

  ASSERT_EQ(
      checkpoint.dictionarySpills().size(), state.dictionarySpills.size());
  for (size_t i = 0; i < state.dictionarySpills.size(); ++i) {
    SCOPED_TRACE(fmt::format("dictionary spill {}", i));
    const auto& spill = checkpoint.dictionarySpills()[i];
    EXPECT_EQ(spill.dictionaryId, state.dictionarySpills[i].dictionaryId);
    EXPECT_EQ(spill.dataType, state.dictionarySpills[i].dataType);
    EXPECT_EQ(spill.stateVersion, state.dictionarySpills[i].stateVersion);
    EXPECT_EQ(spill.state.uncompressedSize(), 256);
  }
}

TEST(CheckpointTest, rejectsMissingNextStreamOffset) {
  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(serializeCheckpoint(
          kCheckpointVersion,
          /*spills=*/{},
          StateSpec{.nextStreamOffset = 0, .dictionarySpills = {}})),
      "Checkpoint is missing the next stream offset.");
}

TEST(CheckpointTest, rejectsAbsentVersion) {
  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(serializeCheckpoint(/*version=*/0)),
      "Unsupported checkpoint version: 0");
}

TEST(CheckpointTest, rejectsFutureVersion) {
  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(serializeCheckpoint(kCheckpointVersion + 1)),
      "Unsupported checkpoint version");
}

TEST(CheckpointTest, rejectsSpillWithoutUncompressedSize) {
  const std::vector<SpillSpec> spills{
      {.family = 1,
       .name = "nimble.hash.v1",
       .stateVersion = 1,
       .offset = 1'024,
       .size = 512,
       .uncompressedSize = 0}};

  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(serializeCheckpoint(kCheckpointVersion, spills)),
      "Checkpoint section is missing an uncompressed size.");
}

// The checkpoint is read on a resume path where a torn tail is expected, so a
// truncated buffer has to fail the verifier rather than be walked.
TEST(CheckpointTest, rejectsTruncatedBuffer) {
  const auto serialized = serializeCheckpoint(kCheckpointVersion);
  ASSERT_GT(serialized.size(), 4);
  const std::string truncated = serialized.substr(0, serialized.size() / 2);

  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(truncated), "Corrupt checkpoint section.");
}

TEST(CheckpointTest, rejectsEmptyBuffer) {
  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(std::string_view{}),
      "Corrupt checkpoint section.");
}

// An IndexSpill whose name or state was never written passes verification,
// because flatbuffers treats both as optional. The parse has to reject them.
TEST(CheckpointTest, rejectsSpillWithoutName) {
  flatbuffers::FlatBufferBuilder builder;
  const auto state = serialization::CreateMetadataSection(
      builder,
      /*offset=*/64,
      /*size=*/32,
      static_cast<serialization::CompressionType>(CompressionType::Zstd),
      /*uncompressed_size=*/128);
  const std::vector<flatbuffers::Offset<serialization::IndexSpill>> spills{
      serialization::CreateIndexSpill(
          builder, /*family=*/1, /*name=*/0, /*state_version=*/1, state)};
  builder.Finish(
      serialization::CreateCheckpoint(
          builder, kCheckpointVersion, builder.CreateVector(spills)));
  const std::string_view serialized{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};

  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(serialized),
      "Index spill is missing its index name.");
}

TEST(CheckpointTest, rejectsSpillWithoutState) {
  flatbuffers::FlatBufferBuilder builder;
  const auto name = builder.CreateString("nimble.hash.v1");
  const std::vector<flatbuffers::Offset<serialization::IndexSpill>> spills{
      serialization::CreateIndexSpill(
          builder, /*family=*/1, name, /*state_version=*/1, /*state=*/0)};
  builder.Finish(
      serialization::CreateCheckpoint(
          builder, kCheckpointVersion, builder.CreateVector(spills)));
  const std::string_view serialized{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};

  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(serialized),
      "Index spill is missing its state section: nimble.hash.v1");
}

TEST(CheckpointTest, rejectsDictionarySpillWithoutState) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<serialization::DictionarySpill>> spills{
      serialization::CreateDictionarySpill(
          builder,
          /*dictionary_id=*/3,
          /*data_type=*/1,
          /*state_version=*/4,
          /*state=*/0)};
  builder.Finish(
      serialization::CreateCheckpoint(
          builder,
          kCheckpointVersion,
          /*index_spills=*/0,
          /*next_stream_offset=*/1,
          builder.CreateVector(spills)));
  const std::string_view serialized{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};

  NIMBLE_ASSERT_FILE_THROW(
      Checkpoint::deserialize(serialized),
      "Dictionary spill is missing its state section: 3");
}

// Plants a checkpoint directly because the writer path is not implemented.
class CheckpointTabletTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("CheckpointTabletTest");
  }

  // Writes a tablet carrying 'checkpointSection' as its checkpoint section,
  // or a finalized tablet when the section is empty.
  std::shared_ptr<TabletReader> createTablet(
      std::string_view checkpointSection) {
    file_.clear();
    velox::InMemoryWriteFile writeFile{&file_};
    auto tabletWriter = TabletWriter::create(&writeFile, *pool_, {});
    if (!checkpointSection.empty()) {
      tabletWriter->writeOptionalSection(
          std::string{kCheckpointSection}, checkpointSection);
    }
    tabletWriter->close();
    writeFile.close();

    readFile_ =
        std::make_shared<testing::InMemoryTrackableReadFile>(file_, false);
    return TabletReader::create(
        readFile_, pool_.get(), test::makeTestTabletOptions(pool_.get()));
  }

  std::string file_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<testing::InMemoryTrackableReadFile> readFile_;
};

TEST_F(CheckpointTabletTest, finalizedFileHasNoCheckpoint) {
  const auto tablet = createTablet(/*checkpointSection=*/{});
  EXPECT_FALSE(tablet->suspended());
  EXPECT_FALSE(tablet->checkpoint().has_value());
}

TEST_F(CheckpointTabletTest, suspendedFileLoadsCheckpointOnDemand) {
  const std::vector<SpillSpec> spills{
      {.family = 1,
       .name = "nimble.hash.v1",
       .stateVersion = 2,
       .offset = 64,
       .size = 32,
       .uncompressedSize = 128}};

  const auto tablet =
      createTablet(serializeCheckpoint(kCheckpointVersion, spills));
  const auto& checkpointSection =
      tablet->optionalSections().at(std::string{kCheckpointSection});

  readFile_->resetChunks();
  EXPECT_TRUE(tablet->suspended());
  EXPECT_TRUE(readFile_->chunks().empty());

  for (size_t attempt = 0; attempt < 2; ++attempt) {
    SCOPED_TRACE(fmt::format("checkpoint load {}", attempt));
    readFile_->resetChunks();
    const auto checkpoint = tablet->checkpoint();
    ASSERT_TRUE(checkpoint.has_value());
    EXPECT_EQ(checkpoint->version(), kCheckpointVersion);
    ASSERT_EQ(checkpoint->indexSpills().size(), 1);
    EXPECT_EQ(checkpoint->indexSpills().front().name, "nimble.hash.v1");
    EXPECT_EQ(checkpoint->indexSpills().front().stateVersion, 2);

    const auto chunks = readFile_->chunks();
    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks.front().offset, checkpointSection.offset());
    EXPECT_EQ(chunks.front().size, checkpointSection.size());
  }
}

TEST_F(CheckpointTabletTest, corruptCheckpointFailsOnDemand) {
  const auto tablet = createTablet("not a flatbuffer");
  EXPECT_TRUE(tablet->suspended());
  NIMBLE_ASSERT_FILE_THROW(tablet->checkpoint(), "Corrupt checkpoint section.");
}

} // namespace
} // namespace facebook::nimble
