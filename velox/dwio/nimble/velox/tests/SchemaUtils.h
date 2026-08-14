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

#include <gtest/gtest.h>

#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaTypes.h"
#include "velox/type/Type.h"

namespace facebook::nimble::test {

void verifySchemaNodes(
    const std::vector<SchemaNode>& nodes,
    std::vector<nimble::SchemaNode> expected);

void compareSchema(
    const std::vector<SchemaNode>& nodes,
    const std::shared_ptr<const nimble::Type>& root);

std::shared_ptr<nimble::RowTypeBuilder> row(
    nimble::SchemaBuilder& builder,
    std::vector<std::pair<std::string, std::shared_ptr<nimble::TypeBuilder>>>
        children);

std::shared_ptr<nimble::ArrayTypeBuilder> array(
    nimble::SchemaBuilder& builder,
    std::shared_ptr<nimble::TypeBuilder> elements);

std::shared_ptr<nimble::ArrayWithOffsetsTypeBuilder> arrayWithOffsets(
    nimble::SchemaBuilder& builder,
    std::shared_ptr<nimble::TypeBuilder> elements);

std::shared_ptr<nimble::MapTypeBuilder> map(
    nimble::SchemaBuilder& builder,
    std::shared_ptr<nimble::TypeBuilder> keys,
    std::shared_ptr<nimble::TypeBuilder> values);

std::shared_ptr<nimble::SlidingWindowMapTypeBuilder> slidingWindowMap(
    nimble::SchemaBuilder& builder,
    std::shared_ptr<nimble::TypeBuilder> keys,
    std::shared_ptr<nimble::TypeBuilder> values);

class FlatMapChildAdder {
 public:
  void addChild(std::string name) {
    NIMBLE_CHECK(schemaBuilder_, "Flat map child adder is not intialized.");
    typeBuilder_->addChild(name, valueFactory_(*schemaBuilder_));
  }

 private:
  void initialize(
      nimble::SchemaBuilder& schemaBuilder,
      nimble::FlatMapTypeBuilder& typeBuilder,
      std::function<std::shared_ptr<nimble::TypeBuilder>(
          nimble::SchemaBuilder&)> valueFactory) {
    schemaBuilder_ = &schemaBuilder;
    typeBuilder_ = &typeBuilder;
    valueFactory_ = std::move(valueFactory);
  }

  nimble::SchemaBuilder* schemaBuilder_;
  nimble::FlatMapTypeBuilder* typeBuilder_;
  std::function<std::shared_ptr<nimble::TypeBuilder>(nimble::SchemaBuilder&)>
      valueFactory_;

  friend std::shared_ptr<nimble::FlatMapTypeBuilder> flatMap(
      nimble::SchemaBuilder& builder,
      nimble::ScalarKind keyScalarKind,
      std::function<std::shared_ptr<nimble::TypeBuilder>(
          nimble::SchemaBuilder&)> valueFactory,
      FlatMapChildAdder& childAdder);
};

std::shared_ptr<nimble::FlatMapTypeBuilder> flatMap(
    nimble::SchemaBuilder& builder,
    nimble::ScalarKind keyScalarKind,
    std::function<std::shared_ptr<nimble::TypeBuilder>(nimble::SchemaBuilder&)>
        valueFactory,
    FlatMapChildAdder& childAdder);

void schema(
    nimble::SchemaBuilder& builder,
    std::function<void(nimble::SchemaBuilder&)> factory);

#define NIMBLE_SCHEMA(schemaBuilder, types) \
  facebook::nimble::test::schema(           \
      schemaBuilder, [&](nimble::SchemaBuilder& builder) { types; })

#define NIMBLE_ROW(...) facebook::nimble::test::row(builder, __VA_ARGS__)
#define NIMBLE_TINYINT() \
  builder.createScalarTypeBuilder(nimble::ScalarKind::Int8)
#define NIMBLE_SMALLINT() \
  builder.createScalarTypeBuilder(nimble::ScalarKind::Int16)
#define NIMBLE_INTEGER() \
  builder.createScalarTypeBuilder(nimble::ScalarKind::Int32)
#define NIMBLE_BIGINT() \
  builder.createScalarTypeBuilder(nimble::ScalarKind::Int64)
#define NIMBLE_REAL() builder.createScalarTypeBuilder(nimble::ScalarKind::Float)
#define NIMBLE_DOUBLE() \
  builder.createScalarTypeBuilder(nimble::ScalarKind::Double)
#define NIMBLE_BOOLEAN() \
  builder.createScalarTypeBuilder(nimble::ScalarKind::Bool)
#define NIMBLE_STRING() \
  builder.createScalarTypeBuilder(nimble::ScalarKind::String)
#define NIMBLE_BINARY() \
  builder.createScalarTypeBuilder(nimble::ScalarKind::Binary)
#define NIMBLE_TIMESTAMPMICRONANO() \
  builder.createTimestampMicroNanoTypeBuilder()
#define NIMBLE_ARRAY(elements) facebook::nimble::test::array(builder, elements)
#define NIMBLE_OFFSETARRAY(elements) \
  facebook::nimble::test::arrayWithOffsets(builder, elements)
// Both operands allocate schema nodes, and the order in which a function's
// arguments are evaluated is unspecified in C++ (GCC evaluates right to left,
// Clang left to right). Passing them directly would therefore hand out schema
// offsets in a compiler dependent order. Sequence them so keys always take the
// lower offset, which is what the recorded expectations assume.
#define NIMBLE_MAP(keys, values)                                        \
  [&]() {                                                               \
    auto nimbleMapKeys = (keys);                                        \
    auto nimbleMapValues = (values);                                    \
    return facebook::nimble::test::map(                                 \
        builder, std::move(nimbleMapKeys), std::move(nimbleMapValues)); \
  }()
#define NIMBLE_SLIDINGWINDOWMAP(keys, values)                                 \
  [&]() {                                                                     \
    auto nimbleWindowKeys = (keys);                                           \
    auto nimbleWindowValues = (values);                                       \
    return facebook::nimble::test::slidingWindowMap(                          \
        builder, std::move(nimbleWindowKeys), std::move(nimbleWindowValues)); \
  }()
#define NIMBLE_FLATMAP(keyKind, values, adder)                \
  facebook::nimble::test::flatMap(                            \
      builder,                                                \
      nimble::ScalarKind::keyKind,                            \
      [&](nimble::SchemaBuilder& builder) { return values; }, \
      adder)

} // namespace facebook::nimble::test
