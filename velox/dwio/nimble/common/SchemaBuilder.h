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

#include <deque>
#include <optional>
#include <string_view>

#include "folly/container/F14Map.h"
#include "folly/container/F14Set.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"

// SchemaBuilder is used to construct a tablet compatible schema.
// The table schema translates logical type tree to its underlying Nimble
// streams stored in the file. Each stream has a unique offset assigned to it,
// which is later used when storing stream related data in the tablet footer.
// The class supports primitive types (scalar types) and complex types: rows,
// arrays, maps and flat maps.
//
// This class serves two purposes:
// 1. Allow constructing parent schema nodes, before creating their children,
// and later allow attaching children to parent nodes. This is needed because
// writers usually construct their inner state top to bottom (recursively), so
// this allows for a much easier integration with writers.
// 2. When writing an Nimble file, not all inner streams are known ahead of
// time. This is because FlatMaps may append new streams when new data arrives,
// and new keys are encountered. This class abstracts the allocation of tablet
// offsets (indices to streams) and makes sure they stay consistent across
// stripes, even while the schema keeps changing.
// NOTE: The offsets allocated by this class are used only during write time,
// and are different than the final serialized offsets. During footer
// serialization, "build time offsets" (offsets managed by this class) are
// translated to "schema ordered" offsets (rely on the final flattened schema
// tree layout).
namespace facebook::nimble {

class SchemaBuilder;
class Type;
class ScalarTypeBuilder;
class TimestampMicroNanoTypeBuilder;
class ArrayTypeBuilder;
class MapTypeBuilder;
class RowTypeBuilder;
class FlatMapTypeBuilder;
class ArrayWithOffsetsTypeBuilder;
class SlidingWindowMapTypeBuilder;

class StreamContext {
 public:
  StreamContext() = default;
  StreamContext(const StreamContext&) = delete;
  StreamContext(StreamContext&&) = delete;
  StreamContext& operator=(const StreamContext&) = delete;
  StreamContext& operator=(StreamContext&&) = delete;
  virtual ~StreamContext() = default;
};

class StreamDescriptorBuilder : public StreamDescriptor {
 public:
  StreamDescriptorBuilder(offset_size offset, ScalarKind scalarKind)
      : StreamDescriptor(offset, scalarKind) {}

  void setContext(std::unique_ptr<StreamContext>&& context) const {
    context_ = std::move(context);
  }

  template <typename T>
  T* context() const {
    return dynamic_cast<T*>(context_.get());
  }

  template <typename T>
  T* checkedContext() const {
    auto* ptr = dynamic_cast<T*>(context_.get());
    NIMBLE_CHECK_NOT_NULL(
        ptr,
        "StreamDescriptorBuilder context is not set or not of expected type");
    return ptr;
  }

 private:
  mutable std::unique_ptr<StreamContext> context_;
};

class TypeBuilderContext {
 public:
  virtual ~TypeBuilderContext() = default;
};

class TypeBuilder {
 public:
  void setContext(std::unique_ptr<TypeBuilderContext>&& context) const {
    context_ = std::move(context);
  }

  template <typename T>
  const T* context() const {
    return dynamic_cast<T*>(context_.get());
  }

  template <typename T>
  const T* checkedContext() const {
    auto* ptr = dynamic_cast<T*>(context_.get());
    NIMBLE_CHECK_NOT_NULL(
        ptr, "TypeBuilder context is not set or not of expected type");
    return ptr;
  }

  Kind kind() const;

  // Sets the per-type string-keyed attribute bag. Insertion order is
  // preserved end-to-end through schema serialization. Overwrites any
  // previously set attributes. Intended for writer-side callers that annotate
  // nodes with format-specific metadata.
  void setAttributes(
      std::vector<std::pair<std::string, std::string>> attributes) {
    attributes_ = std::move(attributes);
  }

  // Returns the per-type string-keyed attribute bag. Empty by default.
  const std::vector<std::pair<std::string, std::string>>& attributes() const {
    return attributes_;
  }

  ScalarTypeBuilder& asScalar();
  TimestampMicroNanoTypeBuilder& asTimestampMicroNano();
  ArrayTypeBuilder& asArray();
  MapTypeBuilder& asMap();
  SlidingWindowMapTypeBuilder& asSlidingWindowMap();
  RowTypeBuilder& asRow();
  FlatMapTypeBuilder& asFlatMap();
  ArrayWithOffsetsTypeBuilder& asArrayWithOffsets();
  const ScalarTypeBuilder& asScalar() const;
  const TimestampMicroNanoTypeBuilder& asTimestampMicroNano() const;
  const ArrayTypeBuilder& asArray() const;
  const MapTypeBuilder& asMap() const;
  const SlidingWindowMapTypeBuilder& asSlidingWindowMap() const;
  const RowTypeBuilder& asRow() const;
  const FlatMapTypeBuilder& asFlatMap() const;
  const ArrayWithOffsetsTypeBuilder& asArrayWithOffsets() const;

 protected:
  TypeBuilder(SchemaBuilder& schemaBuilder, Kind kind);
  virtual ~TypeBuilder() = default;

  SchemaBuilder& schemaBuilder_;

 private:
  Kind kind_;
  mutable std::unique_ptr<TypeBuilderContext> context_;
  std::vector<std::pair<std::string, std::string>> attributes_;

  friend class SchemaBuilder;
};

class ScalarTypeBuilder : public TypeBuilder {
 public:
  const StreamDescriptorBuilder& scalarDescriptor() const;

 private:
  ScalarTypeBuilder(SchemaBuilder& schemaBuilder, ScalarKind scalarKind);

  StreamDescriptorBuilder scalarDescriptor_;

  friend class SchemaBuilder;
};

class TimestampMicroNanoTypeBuilder : public TypeBuilder {
 public:
  const StreamDescriptorBuilder& microsDescriptor() const;
  const StreamDescriptorBuilder& nanosDescriptor() const;

 private:
  explicit TimestampMicroNanoTypeBuilder(SchemaBuilder& schemaBuilder);

  StreamDescriptorBuilder microsDescriptor_;
  StreamDescriptorBuilder nanosDescriptor_;

  friend class SchemaBuilder;
};

class LengthsTypeBuilder : public TypeBuilder {
 public:
  const StreamDescriptorBuilder& lengthsDescriptor() const;

 protected:
  LengthsTypeBuilder(SchemaBuilder& schemaBuilder, Kind kind);

 private:
  StreamDescriptorBuilder lengthsDescriptor_;
};

class ArrayTypeBuilder : public LengthsTypeBuilder {
 public:
  const TypeBuilder& elements() const;
  void setChildren(std::shared_ptr<TypeBuilder> elements);

 private:
  explicit ArrayTypeBuilder(SchemaBuilder& schemaBuilder);

  std::shared_ptr<const TypeBuilder> elements_;

  friend class SchemaBuilder;
};

class MapTypeBuilder : public LengthsTypeBuilder {
 public:
  const TypeBuilder& keys() const;
  const TypeBuilder& values() const;

  void setChildren(
      std::shared_ptr<TypeBuilder> keys,
      std::shared_ptr<TypeBuilder> values);

 private:
  explicit MapTypeBuilder(SchemaBuilder& schemaBuilder);

  std::shared_ptr<const TypeBuilder> keys_;
  std::shared_ptr<const TypeBuilder> values_;

  friend class SchemaBuilder;
};

class SlidingWindowMapTypeBuilder : public TypeBuilder {
 public:
  const StreamDescriptorBuilder& offsetsDescriptor() const;
  const StreamDescriptorBuilder& lengthsDescriptor() const;
  const TypeBuilder& keys() const;
  const TypeBuilder& values() const;

  void setChildren(
      std::shared_ptr<TypeBuilder> keys,
      std::shared_ptr<TypeBuilder> values);

 private:
  explicit SlidingWindowMapTypeBuilder(SchemaBuilder& schemaBuilder);

  StreamDescriptorBuilder offsetsDescriptor_;
  StreamDescriptorBuilder lengthsDescriptor_;
  std::shared_ptr<const TypeBuilder> keys_;
  std::shared_ptr<const TypeBuilder> values_;

  friend class SchemaBuilder;
};

class RowTypeBuilder : public TypeBuilder {
 public:
  const StreamDescriptorBuilder& nullsDescriptor() const;
  size_t childrenCount() const;
  const TypeBuilder& childAt(size_t index) const;
  /// Returns child with name, or throws if no child has that name.
  const TypeBuilder& findChild(std::string_view name) const;
  const std::string& nameAt(size_t index) const;
  void addChild(std::string name, std::shared_ptr<TypeBuilder> child);

 private:
  RowTypeBuilder(SchemaBuilder& schemaBuilder, size_t childrenCount);

  StreamDescriptorBuilder nullsDescriptor_;
  std::vector<std::string> names_;
  std::vector<std::shared_ptr<const TypeBuilder>> children_;

  friend class SchemaBuilder;
};

class FlatMapTypeBuilder : public TypeBuilder {
 public:
  const StreamDescriptorBuilder& nullsDescriptor() const;
  const StreamDescriptorBuilder& inMapDescriptorAt(size_t index) const;
  /// Returns the number of in-map stream descriptors tracked for flat-map keys.
  /// This should match childrenCount(): the schema keeps one descriptor per
  /// key, independent of whether a stripe stores data for that stream.
  size_t inMapDescriptorCount() const;
  size_t childrenCount() const;
  const TypeBuilder& childAt(size_t index) const;
  const std::string& nameAt(size_t index) const;
  ScalarKind keyScalarKind() const;

  const StreamDescriptorBuilder& addChild(
      std::string name,
      std::shared_ptr<TypeBuilder> child);

 private:
  FlatMapTypeBuilder(SchemaBuilder& schemaBuilder, ScalarKind keyScalarKind);

  ScalarKind keyScalarKind_;
  StreamDescriptorBuilder nullsDescriptor_;
  // Uses deque for pointer stability: FlatMapFieldWriter stores StringView
  // keys pointing at these strings, so they must not be invalidated on
  // push_back.
  std::deque<std::string> names_;
  std::vector<std::unique_ptr<StreamDescriptorBuilder>> inMapDescriptors_;
  std::vector<std::shared_ptr<const TypeBuilder>> children_;

  friend class SchemaBuilder;
};

class ArrayWithOffsetsTypeBuilder : public TypeBuilder {
 public:
  const StreamDescriptorBuilder& offsetsDescriptor() const;
  const StreamDescriptorBuilder& lengthsDescriptor() const;
  const TypeBuilder& elements() const;
  void setChildren(std::shared_ptr<TypeBuilder> elements);

 private:
  explicit ArrayWithOffsetsTypeBuilder(SchemaBuilder& schemaBuilder);

  StreamDescriptorBuilder offsetsDescriptor_;
  StreamDescriptorBuilder lengthsDescriptor_;
  std::shared_ptr<const TypeBuilder> elements_;

  friend class SchemaBuilder;
};

class SchemaBuilder {
 public:
  // Create a builder representing a scalar type with kind |scalarKind|.
  std::shared_ptr<ScalarTypeBuilder> createScalarTypeBuilder(
      ScalarKind scalarKind);

  // Create a builder representing a timestamp with microsecond precision and
  // optional nanosecond precision.
  std::shared_ptr<TimestampMicroNanoTypeBuilder>
  createTimestampMicroNanoTypeBuilder();

  // Create an array builder
  std::shared_ptr<ArrayTypeBuilder> createArrayTypeBuilder();

  // Create an array with offsets builder
  std::shared_ptr<ArrayWithOffsetsTypeBuilder>
  createArrayWithOffsetsTypeBuilder();

  // Create a row (struct) builder, with fixed number of children
  // |childrenCount|.
  std::shared_ptr<RowTypeBuilder> createRowTypeBuilder(size_t childrenCount);

  // Create a map builder.
  std::shared_ptr<MapTypeBuilder> createMapTypeBuilder();

  // Create a slidingwindowmap builder.
  std::shared_ptr<SlidingWindowMapTypeBuilder>
  createSlidingWindowMapTypeBuilder();

  // Create a flat map builder. |keyScalarKind| captures the type of the map
  // key.
  std::shared_ptr<FlatMapTypeBuilder> createFlatMapTypeBuilder(
      ScalarKind keyScalarKind);

  // Retrieves all the nodes CURRENTLY known to the schema builder.
  // If more nodes are added to the schema builder later on, following calls to
  // this method will return existing and new nodes. It is guaranteed that the
  // allocated offsets for each node will stay the same across invocations.
  // NOTE: The schema must be in a valid state when calling this method. Valid
  // state means that the schema is a valid tree, with a single root node. This
  // means that all created builders were attached to their parents by calling
  // addChild/setChildren.
  std::vector<SchemaNode> schemaNodes() const;

  offset_size nodeCount() const;

  /// Creates a stripe dictionary stream associated with a value stream.
  offset_size createSharedDictionaryStream(offset_size valueStreamOffset);

  /// Returns the stripe dictionary stream associated with a value stream.
  std::optional<offset_size> sharedDictionaryStreamOffset(
      offset_size valueStreamOffset) const;

  const std::shared_ptr<const TypeBuilder>& root() const;

 private:
  void registerChild(const std::shared_ptr<TypeBuilder>& type);
  offset_size allocateStreamOffset();

  // Schema builder is building a tree of types. As a tree, it should have a
  // single root. We use |roots_| to track that the callers don't forget to
  // attach every created type to a parent (and thus constructing a valid tree).
  // Every time a new type is constructed (and still wasn't attached to a
  // parent), we add it to |roots_|, to indicate that a new root (unattached
  // node) exists. Every time a node is attached to a parent, we remove it from
  // |roots_|, as it is no longer a free standing node.
  // We also use |roots_| to identify two other misuses:
  // 1. If a node was created using a different schema builder instance and then
  // was attached to a parent of a different schema builder.
  // 2. Attaching a node more than once to a parent.
  folly::F14FastSet<std::shared_ptr<const TypeBuilder>> roots_;
  // Used by the physical layout planner to colocate stripe dictionaries with
  // their value streams. File and external dictionaries have no stream here.
  folly::F14FastMap<offset_size, offset_size> sharedDictionaryStreamOffsets_;
  offset_size currentOffset_{0};

  friend class ScalarTypeBuilder;
  friend class TimestampMicroNanoTypeBuilder;
  friend class LengthsTypeBuilder;
  friend class ArrayTypeBuilder;
  friend class ArrayWithOffsetsTypeBuilder;
  friend class RowTypeBuilder;
  friend class MapTypeBuilder;
  friend class FlatMapTypeBuilder;
  friend class SlidingWindowMapTypeBuilder;
  friend std::ostream& operator<<(
      std::ostream& out,
      const SchemaBuilder& schema);
};

std::ostream& operator<<(std::ostream& out, const SchemaBuilder& schema);

/// Converts a deserialized type tree into the same DFS schema node layout
/// produced by SchemaBuilder::schemaNodes().
std::vector<SchemaNode> schemaNodes(const Type& type);

} // namespace facebook::nimble
