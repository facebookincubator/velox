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

#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/SchemaTypes.h"

// Schema reader provides a strongly typed, tree like, reader friendly facade on
// top of a flat tablet schema.
// Schema stored in a tablet footer is a DFS representation of a type tree.
// Reconstructing a tree out of this flat representation require deep knowledge
// of how each complex type (rows, arrays, maps and flat maps) are laid out.
// Using this class, it is possible to reconstruct an easy to use tree
// representation on the logical type tree.
// The main usage of this class is to allow efficient retrieval of subsets of
// the schema tree (used in column and feature projection), as it can
// efficiently skip full sections of the schema tree (which cannot be done
// efficiently using a flat schema representation, that relies on schema node
// order for decoding).
namespace facebook::nimble {

class ScalarType;
class TimestampMicroNanoType;
class RowType;
class ArrayType;
class ArrayWithOffsetsType;
class MapType;
class FlatMapType;
class SlidingWindowMapType;

class Type {
 public:
  Kind kind() const;

  bool isScalar() const;
  bool isTimestampMicroNano() const;
  bool isRow() const;
  bool isArray() const;
  bool isArrayWithOffsets() const;
  bool isMap() const;
  bool isFlatMap() const;
  bool isSlidingWindowMap() const;

  const ScalarType& asScalar() const;
  const TimestampMicroNanoType& asTimestampMicroNano() const;
  const RowType& asRow() const;
  const ArrayType& asArray() const;
  const ArrayWithOffsetsType& asArrayWithOffsets() const;
  const MapType& asMap() const;
  const FlatMapType& asFlatMap() const;
  const SlidingWindowMapType& asSlidingWindowMap() const;

  /// Returns the per-type string-keyed attribute bag. Insertion order
  /// matches the writer-side `TypeBuilder::attributes()` it was sourced
  /// from. Empty when the underlying SchemaNode had no attributes (covers
  /// both legacy files and present-but-empty lists).
  const std::vector<std::pair<std::string, std::string>>& attributes() const {
    return attributes_;
  }

 protected:
  explicit Type(
      Kind kind,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  virtual ~Type() = default;

 private:
  Type(const Type&) = delete;
  Type(Type&&) = delete;

  Kind kind_;
  std::vector<std::pair<std::string, std::string>> attributes_;
};

class ScalarType : public Type {
 public:
  explicit ScalarType(
      StreamDescriptor scalarDescriptor,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  const StreamDescriptor& scalarDescriptor() const;

 private:
  StreamDescriptor scalarDescriptor_;
};

class TimestampMicroNanoType : public Type {
 public:
  explicit TimestampMicroNanoType(
      StreamDescriptor microsDescriptor,
      StreamDescriptor nanosDescriptor,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  const StreamDescriptor& microsDescriptor() const;
  const StreamDescriptor& nanosDescriptor() const;

 private:
  StreamDescriptor microsDescriptor_;
  StreamDescriptor nanosDescriptor_;
};

class ArrayType : public Type {
 public:
  ArrayType(
      StreamDescriptor lengthsDescriptor,
      std::shared_ptr<const Type> elements,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  const StreamDescriptor& lengthsDescriptor() const;
  const std::shared_ptr<const Type>& elements() const;

 protected:
  StreamDescriptor lengthsDescriptor_;
  std::shared_ptr<const Type> elements_;
};

class MapType : public Type {
 public:
  MapType(
      StreamDescriptor lengthsDescriptor,
      std::shared_ptr<const Type> keys,
      std::shared_ptr<const Type> values,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  const StreamDescriptor& lengthsDescriptor() const;
  const std::shared_ptr<const Type>& keys() const;
  const std::shared_ptr<const Type>& values() const;

 protected:
  StreamDescriptor lengthsDescriptor_;
  std::shared_ptr<const Type> keys_;
  std::shared_ptr<const Type> values_;
};

class SlidingWindowMapType : public virtual Type {
 public:
  SlidingWindowMapType(
      StreamDescriptor offsetsDescriptor,
      StreamDescriptor lengthsDescriptor,
      std::shared_ptr<const Type> keys,
      std::shared_ptr<const Type> values,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  const StreamDescriptor& offsetsDescriptor() const;
  const StreamDescriptor& lengthsDescriptor() const;
  const std::shared_ptr<const Type>& keys() const;
  const std::shared_ptr<const Type>& values() const;

 protected:
  StreamDescriptor offsetsDescriptor_;
  StreamDescriptor lengthsDescriptor_;
  std::shared_ptr<const Type> keys_;
  std::shared_ptr<const Type> values_;
};

class RowType : public Type {
 public:
  RowType(
      StreamDescriptor nullsDescriptor,
      std::vector<std::string> names,
      std::vector<std::shared_ptr<const Type>> children,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  const StreamDescriptor& nullsDescriptor() const;
  size_t childrenCount() const;
  const std::shared_ptr<const Type>& childAt(size_t index) const;
  const std::string& nameAt(size_t index) const;

  const std::vector<std::string>& names() const {
    return names_;
  }

  const std::vector<std::shared_ptr<const Type>>& children() const {
    return children_;
  }

  /// Finds a child by name.
  /// @return Child index if found, std::nullopt otherwise.
  std::optional<size_t> findChild(std::string_view name) const;

 protected:
  StreamDescriptor nullsDescriptor_;
  std::vector<std::string> names_;
  std::vector<std::shared_ptr<const Type>> children_;
};

class FlatMapType : public Type {
 public:
  FlatMapType(
      StreamDescriptor nullsDescriptor,
      ScalarKind keyScalarKind,
      std::vector<std::string> names,
      std::vector<std::unique_ptr<StreamDescriptor>> inMapDescriptors,
      std::vector<std::shared_ptr<const Type>> children,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  const StreamDescriptor& nullsDescriptor() const;
  const StreamDescriptor& inMapDescriptorAt(size_t index) const;
  /// Returns the number of in-map stream descriptors tracked for flat-map keys.
  /// This should match childrenCount(): the schema keeps one descriptor per
  /// key, independent of whether a stripe stores data for that stream.
  size_t inMapDescriptorCount() const;
  ScalarKind keyScalarKind() const;
  size_t childrenCount() const;
  const std::shared_ptr<const Type>& childAt(size_t index) const;
  const std::string& nameAt(size_t index) const;

  /// Finds a child (key) by name.
  /// @return Child index if found, std::nullopt otherwise.
  std::optional<size_t> findChild(std::string_view name) const;

 private:
  StreamDescriptor nullsDescriptor_;
  ScalarKind keyScalarKind_;
  std::vector<std::string> names_;
  std::vector<std::unique_ptr<StreamDescriptor>> inMapDescriptors_;
  std::vector<std::shared_ptr<const Type>> children_;
};

class ArrayWithOffsetsType : public Type {
 public:
  ArrayWithOffsetsType(
      StreamDescriptor offsetsDescriptor,
      StreamDescriptor lengthsDescriptor,
      std::shared_ptr<const Type> elements,
      std::vector<std::pair<std::string, std::string>> attributes = {});

  const StreamDescriptor& offsetsDescriptor() const;
  const StreamDescriptor& lengthsDescriptor() const;
  const std::shared_ptr<const Type>& elements() const;

 protected:
  StreamDescriptor offsetsDescriptor_;
  StreamDescriptor lengthsDescriptor_;
  std::shared_ptr<const Type> elements_;
};

class SchemaReader {
 public:
  // Construct type tree from an ordered list of schema nodes.
  // If the schema nodes are ordered incorrectly the behavior is undefined.
  static std::shared_ptr<const Type> getSchema(
      const std::vector<SchemaNode>& nodes);

  struct NodeInfo {
    std::string_view name;
    const Type* parentType;
    size_t placeInSibling = 0;
  };

  static void traverseSchema(
      const std::shared_ptr<const Type>& root,
      std::function<void(
          uint32_t /* level */,
          const Type& /* streamType */,
          const NodeInfo& /* traverseInfo */)> visitor);
};

std::ostream& operator<<(
    std::ostream& out,
    const std::shared_ptr<const Type>& root);

/// Visits every value-stream anchor offset in the subtree of `type`, calling
/// `visit(offset)` at each anchor leaf. Returns `true` iff any invocation of
/// `visit` returned `true`; visitor returning `true` short-circuits the walk.
///
/// "Anchor leaves" are streams whose presence reliably proves the subtree
/// contains data in the current stripe (e.g. Scalar data stream, Array
/// lengths stream, Map lengths stream). Stream-bearing containers whose
/// top-level stream may be omitted by the writer (Row's nulls, FlatMap's
/// nulls) are NOT visited directly — instead the walk recurses into ALL
/// their children, since any populated child proves the container has data.
///
/// NOTE: Row recurses into ALL children, not just the first. This is a
/// deliberate change from the legacy `hasValueStreams` walker, which
/// short-circuited at Row's first child as an optimization that exploited
/// Nimble's per-row "all-Row-fields-populate-together" writer invariant.
/// Predicate-style callers may therefore probe additional siblings before
/// the visitor returns true, which is harmless given the same writer
/// invariant; collect-style callers receive every leaf, not just the
/// first.
///
/// This is the single primitive for "tell me about this subtree's anchor
/// streams"; callers compose it with whatever leaf action they need:
///   - "does any anchor pass a predicate?"  → predicate returns true to stop.
///   - "collect all anchor offsets"         → visitor pushes to a sink, returns
///   false.
///   - "mark each anchor in a bitmap"       → visitor sets a bit, returns
///   false.
///
/// PERFORMANCE: This is a function template (not a function taking
/// `std::function`/`folly::FunctionRef`) on purpose. The visitor's call type
/// is concrete at the call site, so `visit(offset)` becomes a direct call
/// the compiler can inline. Earlier `std::function`-based variants paid an
/// indirect-call (`_Function_handler::_M_invoke`) per leaf, which showed up
/// as a hotspot in the Deserializer's per-batch flatmap in-map detection
/// over hundreds of keys. Concrete-typed callables remove that dispatch
/// without forcing every caller to materialize an intermediate offset list.
template <typename Visitor>
bool visitValueStreamLeaves(const Type& type, Visitor& visit) {
  switch (type.kind()) {
    case Kind::Scalar:
      return visit(type.asScalar().scalarDescriptor().offset());
    case Kind::TimestampMicroNano:
      return visit(type.asTimestampMicroNano().microsDescriptor().offset());
    case Kind::Array:
      return visit(type.asArray().lengthsDescriptor().offset());
    case Kind::ArrayWithOffsets:
      return visit(type.asArrayWithOffsets().offsetsDescriptor().offset());
    case Kind::Map:
      return visit(type.asMap().lengthsDescriptor().offset());
    case Kind::SlidingWindowMap:
      return visit(type.asSlidingWindowMap().offsetsDescriptor().offset());
    case Kind::Row: {
      const auto& row = type.asRow();
      NIMBLE_CHECK_GT(
          row.childrenCount(), 0, "Row type must have at least one child");
      // Recurse into ALL children. Row's own nulls stream may be omitted by
      // the writer, so it is not a reliable anchor; children are. The
      // visitor's return value short-circuits the walk on the first hit.
      for (size_t i = 0; i < row.childrenCount(); ++i) {
        if (visitValueStreamLeaves(*row.childAt(i), visit)) {
          return true;
        }
      }
      return false;
    }
    case Kind::FlatMap: {
      const auto& flatMap = type.asFlatMap();
      NIMBLE_CHECK_GT(
          flatMap.childrenCount(),
          0,
          "FlatMap type must have at least one child");
      // FlatMap children are independent keys; each may or may not carry
      // data in the current stripe, so all must be visited.
      for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
        if (visitValueStreamLeaves(*flatMap.childAt(i), visit)) {
          return true;
        }
      }
      return false;
    }
    default:
      NIMBLE_UNREACHABLE("Unsupported type kind: {}", type.kind());
  }
}

template <typename Visitor>
bool visitValueStreamLeaves(const Type& type, Visitor&& visit) {
  auto&& visitRef = std::forward<Visitor>(visit);
  return visitValueStreamLeaves(type, visitRef);
}

/// Visits stream offsets whose presence proves `type` has data in the current
/// stripe. Unlike visitValueStreamLeaves(), this includes container presence
/// streams such as Row and FlatMap null streams, and nested FlatMap in-map
/// streams.
template <typename Visitor>
bool visitPresenceStreamOffsets(const Type& type, Visitor& visit) {
  switch (type.kind()) {
    case Kind::Scalar:
      return visit(type.asScalar().scalarDescriptor().offset());
    case Kind::TimestampMicroNano:
      return visit(type.asTimestampMicroNano().microsDescriptor().offset()) ||
          visit(type.asTimestampMicroNano().nanosDescriptor().offset());
    case Kind::Array: {
      const auto& array = type.asArray();
      return visit(array.lengthsDescriptor().offset()) ||
          visitPresenceStreamOffsets(*array.elements(), visit);
    }
    case Kind::ArrayWithOffsets: {
      const auto& array = type.asArrayWithOffsets();
      return visit(array.offsetsDescriptor().offset()) ||
          visit(array.lengthsDescriptor().offset()) ||
          visitPresenceStreamOffsets(*array.elements(), visit);
    }
    case Kind::Map: {
      const auto& map = type.asMap();
      return visit(map.lengthsDescriptor().offset()) ||
          visitPresenceStreamOffsets(*map.keys(), visit) ||
          visitPresenceStreamOffsets(*map.values(), visit);
    }
    case Kind::SlidingWindowMap: {
      const auto& map = type.asSlidingWindowMap();
      return visit(map.offsetsDescriptor().offset()) ||
          visit(map.lengthsDescriptor().offset()) ||
          visitPresenceStreamOffsets(*map.keys(), visit) ||
          visitPresenceStreamOffsets(*map.values(), visit);
    }
    case Kind::Row: {
      const auto& row = type.asRow();
      if (visit(row.nullsDescriptor().offset())) {
        return true;
      }
      for (size_t i = 0; i < row.childrenCount(); ++i) {
        if (visitPresenceStreamOffsets(*row.childAt(i), visit)) {
          return true;
        }
      }
      return false;
    }
    case Kind::FlatMap: {
      const auto& flatMap = type.asFlatMap();
      if (visit(flatMap.nullsDescriptor().offset())) {
        return true;
      }
      for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
        if (visit(flatMap.inMapDescriptorAt(i).offset()) ||
            visitPresenceStreamOffsets(*flatMap.childAt(i), visit)) {
          return true;
        }
      }
      return false;
    }
    default:
      NIMBLE_UNREACHABLE("Unsupported type kind: {}", type.kind());
  }
}

template <typename Visitor>
bool visitPresenceStreamOffsets(const Type& type, Visitor&& visit) {
  auto&& visitRef = std::forward<Visitor>(visit);
  return visitPresenceStreamOffsets(type, visitRef);
}

} // namespace facebook::nimble
