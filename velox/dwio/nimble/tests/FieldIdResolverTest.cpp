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

#include <gtest/gtest.h>

#include "velox/dwio/common/ParquetFieldId.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"
#include "velox/dwio/nimble/reader/FieldIdResolver.h"
#include "velox/type/Type.h"

// ---------------------------------------------------------------------------
// Iceberg V3 field-id resolution for NIMBLE-backed Iceberg files.
//
// `resolveSchema` walks the file's `facebook::nimble::Type`
// tree in lockstep with the requested (table) Velox schema and the per-column
// Iceberg field-id tree, producing two parallel schemas (both the full file
// schema in file column order): an "exposed" schema whose matched columns
// adopt the requested table names (surfaced as the reader's `rowType()` and
// consumed by the connector's by-name null-fill) and a "reading" schema
// carrying the file's on-disk names. These tests exercise the recursive walk
// across top-level columns, nested struct fields, array element fields, and
// map key/value fields; the add/drop/rename schema-evolution cases (added
// columns excluded so the connector null-fills them, dropped columns retained
// for positional alignment); and the structural-mismatch fallback that
// returns std::nullopt and lets the caller fall back to the legacy name-based
// path.
// ---------------------------------------------------------------------------
namespace facebook::velox::nimble {
namespace {

using facebook::nimble::ArrayType;
using facebook::nimble::Kind;
using facebook::nimble::MapType;
using facebook::nimble::RowType;
using facebook::nimble::ScalarKind;
using facebook::nimble::ScalarType;
using facebook::nimble::StreamDescriptor;
using facebook::nimble::Type;

using Attrs = std::vector<std::pair<std::string, std::string>>;

// Builds a unique stream descriptor each call. The offset/scalarKind are
// not exercised by the resolver -- only `attributes()` and the structural
// shape are.
StreamDescriptor descriptor(ScalarKind kind = ScalarKind::Int64) {
  static facebook::nimble::offset_size next{0};
  return StreamDescriptor{next++, kind};
}

// Helper that stamps the `iceberg.id` attribute on a NIMBLE Type. Mirrors
// what `IcebergDataSink` stamps onto the writer side.
Attrs icebergId(int32_t id) {
  return {{"iceberg.id", std::to_string(id)}};
}

// Resolves using the Iceberg "iceberg.id" attribute key -- the concrete key
// the NIMBLE reader wires into the generic FieldIdResolver.
std::optional<FieldIdSchemaResolution> resolveByFieldId(
    const RowTypePtr& requestedSchema,
    const std::vector<dwio::common::ParquetFieldId>& requestedFieldIds,
    const facebook::nimble::Type& fileSchema) {
  return resolveSchema(
      requestedSchema, requestedFieldIds, fileSchema, "iceberg.id");
}

// Anonymous-namespace synonyms for nimble Type subclasses, so the test
// fixtures read close to the prod schema construction at
// `facebook::nimble::SchemaReader::getType`.
std::shared_ptr<const Type> scalar(int32_t icebergFieldId) {
  return std::make_shared<ScalarType>(descriptor(), icebergId(icebergFieldId));
}

std::shared_ptr<const Type> scalarWithoutId() {
  return std::make_shared<ScalarType>(descriptor(), Attrs{});
}

std::shared_ptr<const Type> row(
    int32_t icebergFieldId,
    std::vector<std::string> names,
    std::vector<std::shared_ptr<const Type>> children) {
  return std::make_shared<RowType>(
      descriptor(),
      std::move(names),
      std::move(children),
      icebergId(icebergFieldId));
}

std::shared_ptr<const Type> rowWithoutId(
    std::vector<std::string> names,
    std::vector<std::shared_ptr<const Type>> children) {
  return std::make_shared<RowType>(
      descriptor(), std::move(names), std::move(children), Attrs{});
}

std::shared_ptr<const Type> array(
    int32_t icebergFieldId,
    std::shared_ptr<const Type> elements) {
  return std::make_shared<ArrayType>(
      descriptor(), std::move(elements), icebergId(icebergFieldId));
}

std::shared_ptr<const Type> map(
    int32_t icebergFieldId,
    std::shared_ptr<const Type> keys,
    std::shared_ptr<const Type> values) {
  return std::make_shared<MapType>(
      descriptor(),
      std::move(keys),
      std::move(values),
      icebergId(icebergFieldId));
}

// Top-level file Row -- the schema-root analog of what
// `BatchReader::schema()` returns. Carries no `iceberg.id` attribute at
// the root (matches the writer side, which only stamps attributes on
// columns, not the synthetic schema-root Row).
std::shared_ptr<const Type> fileSchema(
    std::vector<std::string> names,
    std::vector<std::shared_ptr<const Type>> children) {
  return rowWithoutId(std::move(names), std::move(children));
}

dwio::common::ParquetFieldId pf(
    int32_t fieldId,
    std::vector<dwio::common::ParquetFieldId> children = {}) {
  return dwio::common::ParquetFieldId{
      .fieldId = fieldId, .children = std::move(children)};
}

class FieldIdResolverTest : public ::testing::Test {};

// Top-level rename. File has columns ("renamed_a", "b") with ids 100,
// 200. Caller requests ("a", "b") in original order with ids 100, 200.
// The exposed schema (consumed by the connector by table name) carries
// ("a", "b"); the reading schema (file's on-disk names) carries
// ("renamed_a", "b"). Both are in file column order.
TEST_F(FieldIdResolverTest, topLevelRename) {
  auto file = fileSchema(
      {"renamed_a", "b"}, {scalar(/*icebergFieldId=*/100), scalar(200)});
  auto requested = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  auto resolved =
      resolveByFieldId(requested, {pf(/*fieldId=*/100), pf(200)}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->names(),
      std::vector<std::string>({"a", "b"}));
  EXPECT_EQ(
      resolved->fileNamedSchema->names(),
      std::vector<std::string>({"renamed_a", "b"}));
}

// Nested struct rename. File has user{renamed_name VARCHAR, age INT}
// where `name` was renamed to `renamed_name`. Caller requests
// user{name VARCHAR, age INT}. The exposed nested Row carries the table
// names; the reading nested Row carries the file names.
TEST_F(FieldIdResolverTest, nestedStructFieldRename) {
  auto file = fileSchema(
      {"user"},
      {row(
          /*icebergFieldId=*/100,
          {"renamed_name", "age"},
          {scalar(101), scalar(102)})});
  auto requested =
      ROW({"user"}, {ROW({"name", "age"}, {VARCHAR(), INTEGER()})});
  auto resolved =
      resolveByFieldId(requested, {pf(100, {pf(101), pf(102)})}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->childAt(0)->asRow().names(),
      std::vector<std::string>({"name", "age"}));
  EXPECT_EQ(
      resolved->fileNamedSchema->childAt(0)->asRow().names(),
      std::vector<std::string>({"renamed_name", "age"}));
}

// ARRAY of ROW where the element struct's field was renamed. File has
// items: array<{renamed_label VARCHAR, count INT}>. Caller requests
// items: array<{label VARCHAR, count INT}>. The walk recurses through
// the array's element into the nested Row and rewrites the child name.
TEST_F(FieldIdResolverTest, arrayOfRowWithRenamedElementField) {
  auto elementInFile = row(/*icebergFieldId=*/101,
                           {"renamed_label", "count"},
                           {scalar(102), scalar(103)});
  auto file = fileSchema(
      {"items"}, {array(/*icebergFieldId=*/100, std::move(elementInFile))});
  auto requested =
      ROW({"items"}, {ARRAY(ROW({"label", "count"}, {VARCHAR(), INTEGER()}))});
  auto resolved = resolveByFieldId(
      requested, {pf(100, {pf(101, {pf(102), pf(103)})})}, *file);
  ASSERT_TRUE(resolved.has_value());
  const auto& exposedElement =
      resolved->tableNamedSchema->childAt(0)->asArray().elementType()->asRow();
  EXPECT_EQ(
      exposedElement.names(), std::vector<std::string>({"label", "count"}));
  const auto& readingElement =
      resolved->fileNamedSchema->childAt(0)->asArray().elementType()->asRow();
  EXPECT_EQ(
      readingElement.names(),
      std::vector<std::string>({"renamed_label", "count"}));
}

// MAP<INT, ROW> where the value struct's field was renamed. File has
// scores: map<int, {renamed_score DOUBLE, units VARCHAR}>. Caller
// requests map<int, {score DOUBLE, units VARCHAR}>. The walk recurses
// through both the map's key (scalar) and value (Row).
TEST_F(FieldIdResolverTest, mapIntToRowWithRenamedValueField) {
  auto valueInFile = row(/*icebergFieldId=*/102,
                         {"renamed_score", "units"},
                         {scalar(103), scalar(104)});
  auto file = fileSchema(
      {"scores"},
      {map(/*icebergFieldId=*/100,
           scalar(/*key icebergFieldId=*/101),
           std::move(valueInFile))});
  auto requested =
      ROW({"scores"},
          {MAP(INTEGER(), ROW({"score", "units"}, {DOUBLE(), VARCHAR()}))});
  auto resolved = resolveByFieldId(
      requested, {pf(100, {pf(101), pf(102, {pf(103), pf(104)})})}, *file);
  ASSERT_TRUE(resolved.has_value());
  const auto& exposedValue =
      resolved->tableNamedSchema->childAt(0)->asMap().valueType()->asRow();
  EXPECT_EQ(exposedValue.names(), std::vector<std::string>({"score", "units"}));
  const auto& readingValue =
      resolved->fileNamedSchema->childAt(0)->asMap().valueType()->asRow();
  EXPECT_EQ(
      readingValue.names(),
      std::vector<std::string>({"renamed_score", "units"}));
}

// Two-level nesting: array<struct<inner_array<scalar>>>. Verifies the
// recursion descends past two intervening complex types and renames the
// nested Row field in the exposed schema while the reading schema keeps
// the file name.
TEST_F(FieldIdResolverTest, twoLevelArrayOfRowOfArray) {
  auto innermost = scalar(/*icebergFieldId=*/104);
  auto innerArray = array(/*icebergFieldId=*/103, std::move(innermost));
  auto innerRow =
      row(/*icebergFieldId=*/102, {"renamed_inner"}, {std::move(innerArray)});
  auto outerArray = array(/*icebergFieldId=*/101, std::move(innerRow));
  auto file = fileSchema({"outer"}, {std::move(outerArray)});
  auto requested = ROW({"outer"}, {ARRAY(ROW({"inner"}, {ARRAY(BIGINT())}))});
  auto resolved = resolveByFieldId(
      requested, {pf(101, {pf(102, {pf(103, {pf(104)})})})}, *file);
  ASSERT_TRUE(resolved.has_value());
  const auto& exposedInner =
      resolved->tableNamedSchema->childAt(0)->asArray().elementType()->asRow();
  EXPECT_EQ(exposedInner.names(), std::vector<std::string>({"inner"}));
  const auto& readingInner =
      resolved->fileNamedSchema->childAt(0)->asArray().elementType()->asRow();
  EXPECT_EQ(readingInner.names(), std::vector<std::string>({"renamed_inner"}));
}

// Add column at the top level (the core fix). File has ("a", "b") with
// ids 1, 2. Caller requests ("a", "b", "c") with ids 1, 2, 3. The added
// column `c` is absent from the file, so it appears in neither schema --
// the connector null-fills it downstream. The exposed/reading schemas
// carry only the file's columns, in file order.
TEST_F(FieldIdResolverTest, addColumnTopLevel) {
  auto file = fileSchema({"a", "b"}, {scalar(1), scalar(2)});
  auto requested = ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), INTEGER()});
  auto resolved = resolveByFieldId(requested, {pf(1), pf(2), pf(3)}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->names(),
      std::vector<std::string>({"a", "b"}));
  EXPECT_EQ(
      resolved->fileNamedSchema->names(), std::vector<std::string>({"a", "b"}));
}

// Add column inside a nested struct. File has user{a INT}; caller
// requests user{a INT, b INT}. The added nested field `b` is absent from
// the file, so the exposed nested Row carries only {a}.
TEST_F(FieldIdResolverTest, addColumnNested) {
  auto file =
      fileSchema({"user"}, {row(/*icebergFieldId=*/100, {"a"}, {scalar(101)})});
  auto requested = ROW({"user"}, {ROW({"a", "b"}, {INTEGER(), INTEGER()})});
  auto resolved =
      resolveByFieldId(requested, {pf(100, {pf(101), pf(102)})}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->childAt(0)->asRow().names(),
      std::vector<std::string>({"a"}));
}

// Drop column. File has ("a", "b", "c") with ids 1, 2, 3. Caller
// requests ("a", "c") -- `b` was dropped. The dropped column must be
// RETAINED (under its file name) in both schemas so the reader's
// positional column walk stays aligned with the physical tablet; it is
// simply never selected or projected. Connector matching by table name
// ("a", "c") still resolves correctly.
TEST_F(FieldIdResolverTest, dropColumnRetainsFileColumn) {
  auto file = fileSchema({"a", "b", "c"}, {scalar(1), scalar(2), scalar(3)});
  auto requested = ROW({"a", "c"}, {BIGINT(), BIGINT()});
  auto resolved = resolveByFieldId(requested, {pf(1), pf(3)}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->names(),
      std::vector<std::string>({"a", "b", "c"}));
  EXPECT_EQ(
      resolved->fileNamedSchema->names(),
      std::vector<std::string>({"a", "b", "c"}));
}

// Drop + re-add same name. File has ("c") id 1. Caller requests ("c") id 9 --
// the original c was dropped and a new c with a different id was added. The
// dropped file column must NOT be exposed under the name "c" (which would
// shadow the re-added column in the connector's by-name null-fill); it is
// renamed to a reserved sentinel. The re-added c is absent from the exposed
// schema, so the connector null-fills it.
TEST_F(FieldIdResolverTest, dropReaddColumnSameName) {
  auto file = fileSchema({"c"}, {scalar(1)});
  auto requested = ROW({"c"}, {INTEGER()});
  auto resolved = resolveByFieldId(requested, {pf(9)}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->names(),
      std::vector<std::string>({"$nimble_dropped_0"}));
  EXPECT_EQ(
      resolved->fileNamedSchema->names(), std::vector<std::string>({"c"}));
}

// Rename + add combined. File has ("old_a", "b") ids 1, 2; caller
// requests ("a", "b", "c") ids 1, 2, 3. `old_a` is renamed to `a`, `c`
// is added (absent from file). Exposed carries ("a", "b"); reading
// carries ("old_a", "b").
TEST_F(FieldIdResolverTest, renameAndAddCombined) {
  auto file = fileSchema({"old_a", "b"}, {scalar(1), scalar(2)});
  auto requested = ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), INTEGER()});
  auto resolved = resolveByFieldId(requested, {pf(1), pf(2), pf(3)}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->names(),
      std::vector<std::string>({"a", "b"}));
  EXPECT_EQ(
      resolved->fileNamedSchema->names(),
      std::vector<std::string>({"old_a", "b"}));
}

// Add + rename + drop all at once. File has ("old_a", "dropped", "c")
// ids 1, 2, 3. Caller requests ("a", "c", "d") ids 1, 3, 4: `old_a`
// renamed to `a`, id-2 column dropped, `c` kept, `d` added. The exposed
// schema is file order with `old_a`->`a`, the dropped column retained
// under its file name, and `c` kept; `d` is absent (null-filled by the
// connector). The reading schema is the file's on-disk names.
TEST_F(FieldIdResolverTest, addRenameDropAllAtOnce) {
  auto file =
      fileSchema({"old_a", "dropped", "c"}, {scalar(1), scalar(2), scalar(3)});
  auto requested = ROW({"a", "c", "d"}, {BIGINT(), BIGINT(), INTEGER()});
  auto resolved = resolveByFieldId(requested, {pf(1), pf(3), pf(4)}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->names(),
      std::vector<std::string>({"a", "dropped", "c"}));
  EXPECT_EQ(
      resolved->fileNamedSchema->names(),
      std::vector<std::string>({"old_a", "dropped", "c"}));
}

// Kind mismatch: file column is a scalar but caller requests it as a
// ROW. The walk must bail with std::nullopt so the caller falls back to
// the legacy name-based path.
TEST_F(FieldIdResolverTest, kindMismatchReturnsNullopt) {
  auto file = fileSchema({"renamed_a"}, {scalar(/*icebergFieldId=*/100)});
  auto requested = ROW({"a"}, {ROW({"x"}, {BIGINT()})});
  auto resolved = resolveByFieldId(requested, {pf(100, {pf(101)})}, *file);
  EXPECT_FALSE(resolved.has_value());
}

// A file column without an `iceberg.id` cannot be matched by field-id, so
// it is treated as an unmatched (kept) file column rather than bailing.
// The requested id-100 column is treated as an add (null-filled by the
// connector). This is the add-column fix: a missing id no longer disables
// field-id resolution for the whole row.
TEST_F(FieldIdResolverTest, fileColumnWithoutIcebergIdKept) {
  auto file = fileSchema({"a"}, {scalarWithoutId()});
  auto requested = ROW({"a"}, {BIGINT()});
  auto resolved = resolveByFieldId(requested, {pf(/*fieldId=*/100)}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->names(), std::vector<std::string>({"a"}));
  EXPECT_EQ(
      resolved->fileNamedSchema->names(), std::vector<std::string>({"a"}));
}

// A nested struct child without an `iceberg.id` is kept (under its file
// name) rather than bailing; the requested id-101 nested column is an add.
TEST_F(FieldIdResolverTest, nestedFileColumnWithoutIcebergIdKept) {
  auto file = fileSchema(
      {"user"},
      {row(/*icebergFieldId=*/100,
           {"name", "age"},
           {scalarWithoutId(), scalar(102)})});
  auto requested =
      ROW({"user"}, {ROW({"name", "age"}, {VARCHAR(), INTEGER()})});
  auto resolved =
      resolveByFieldId(requested, {pf(100, {pf(101), pf(102)})}, *file);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(
      resolved->tableNamedSchema->childAt(0)->asRow().names(),
      std::vector<std::string>({"name", "age"}));
}

// ParquetFieldId children count mismatch at MAP level: a MAP must have
// exactly two children in [key, value] order. Anything else bails.
TEST_F(FieldIdResolverTest, mapFieldIdChildrenCountMismatch) {
  auto file = fileSchema(
      {"scores"}, {map(/*icebergFieldId=*/100, scalar(101), scalar(102))});
  auto requested = ROW({"scores"}, {MAP(INTEGER(), BIGINT())});
  // Only 1 child instead of [key, value]. Walk bails.
  auto resolved = resolveByFieldId(requested, {pf(100, {pf(101)})}, *file);
  EXPECT_FALSE(resolved.has_value());
}

// Requested schema / fieldId count mismatch at the top level. Top-level
// is wrapped in a synthesized ParquetFieldId, so the size check happens
// inside the Row dispatch arm against `requestedRow.size()`.
TEST_F(FieldIdResolverTest, topLevelChildrenCountMismatch) {
  auto file = fileSchema({"a", "b"}, {scalar(100), scalar(200)});
  auto requested = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  // Only one fieldId supplied for two requested columns.
  auto resolved = resolveByFieldId(requested, {pf(/*fieldId=*/100)}, *file);
  EXPECT_FALSE(resolved.has_value());
}

} // namespace
} // namespace facebook::velox::nimble
