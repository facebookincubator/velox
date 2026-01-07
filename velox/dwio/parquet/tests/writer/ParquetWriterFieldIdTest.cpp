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

#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/tests/FileReader.h"

namespace {

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

class ParquetWriterFieldIdTest : public ParquetTestBase,
                                 public ::testing::WithParamInterface<bool> {};

TEST_P(ParquetWriterFieldIdTest, fieldIds) {
  auto schema =
      ROW({"p", "s", "a", "m"},
          {BIGINT(),
           ROW({"x", "y"}, {INTEGER(), VARCHAR()}),
           ARRAY(INTEGER()),
           MAP(VARCHAR(), INTEGER())});
  constexpr int32_t kRows = 10;
  auto data = makeRowVector(
      {"p", "s", "a", "m"},
      {makeFlatVector<int64_t>(kRows, [](auto row) { return row; }),
       makeRowVector(
           {"x", "y"},
           {makeFlatVector<int32_t>(kRows, [](auto row) { return row; }),
            makeFlatVector<std::string>(kRows, [](auto) { return "z"; })}),
       makeArrayVectorFromJson<int32_t>(std::vector<std::string>(kRows, "[3]")),
       makeMapVectorFromJson<std::string, int32_t>(
           std::vector<std::string>(kRows, R"({"k": 4})"))});

  parquet::WriterOptions writerOptions;
  writerOptions.memoryPool = rootPool_.get();

  if (GetParam()) {
    // Provide Parquet field IDs aligned with the Velox schema tree.
    // p -> 10.
    // s -> 20, children: x -> 21, y -> 22.
    // a -> 30, list element -> 31.
    // m -> 40, children: key -> 41, value -> 42.
    writerOptions.parquetFieldIds = {
        ParquetFieldId{10, {}},
        ParquetFieldId{20, {ParquetFieldId{21, {}}, ParquetFieldId{22, {}}}},
        ParquetFieldId{30, {ParquetFieldId{31, {}}}},
        ParquetFieldId{40, {ParquetFieldId{41, {}}, ParquetFieldId{42, {}}}},
    };
  }

  auto* sinkPtr = write(data, writerOptions);

  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  auto parquetReader = createReaderInMemory(*sinkPtr, readerOptions);
  EXPECT_EQ(parquetReader->numberOfRows(), kRows);
  auto veloxRowType = parquetReader->rowType();
  EXPECT_EQ(*veloxRowType, *schema);

  std::string_view sinkData(sinkPtr->data(), sinkPtr->size());
  auto arrowBufferReader = std::make_shared<::arrow::io::BufferReader>(
      std::make_shared<::arrow::Buffer>(
          reinterpret_cast<const uint8_t*>(sinkData.data()), sinkData.size()));

  auto fileReader = parquet::arrow::ParquetFileReader::Open(arrowBufferReader);
  auto metadata = fileReader->metadata();
  auto* descr = metadata->schema();
  auto* root = descr->group_node();

  ASSERT_EQ(root->field_count(), 4);

  auto exp = [&](int32_t expectedFieldId) {
    return GetParam() ? expectedFieldId : -1;
  };

  // Top-level field IDs.
  EXPECT_EQ(root->field(0)->field_id(), exp(10));
  EXPECT_EQ(root->field(1)->field_id(), exp(20));
  EXPECT_EQ(root->field(2)->field_id(), exp(30));
  EXPECT_EQ(root->field(3)->field_id(), exp(40));

  using GroupNode = parquet::arrow::schema::GroupNode;
  auto* s = static_cast<const GroupNode*>(root->field(1).get());
  EXPECT_EQ(s->field(0)->field_id(), exp(21));
  EXPECT_EQ(s->field(1)->field_id(), exp(22));

  auto* a = static_cast<const GroupNode*>(root->field(2).get());
  // LIST logical group has one repeated child (the array entries); dive once
  // more to the element.
  auto* listEntries = a->field(0).get();
  auto* listGroup = static_cast<const GroupNode*>(listEntries);
  auto* element = listGroup->field(0).get();
  EXPECT_EQ(element->field_id(), exp(31));

  auto* m = static_cast<const GroupNode*>(root->field(3).get());
  auto* keyValue = m->field(0).get();
  auto* keyValueGroup = static_cast<const GroupNode*>(keyValue);
  EXPECT_EQ(keyValueGroup->field(0)->field_id(), exp(41));
  EXPECT_EQ(keyValueGroup->field(1)->field_id(), exp(42));
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    ParquetWriterFieldIdTest,
    ParquetWriterFieldIdTest,
    ::testing::Values(false, true));

} // namespace
