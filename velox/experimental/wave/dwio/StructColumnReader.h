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

#include "velox/experimental/wave/dwio/ColumnReader.h"

namespace facebook ::velox::wave {

/// Reader of a struct. The children are read as separate columns. A format
/// makes the children in the constructor of its subclass with addChildren().
class StructColumnReader : public ColumnReader {
 public:
  using Path = std::vector<std::unique_ptr<common::Subfield::PathElement>>;

  StructColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      AbstractOperand* operand,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec,
      bool isRoot)
      : ColumnReader(requestedType, fileType, operand, params, scanSpec),
        isRoot_(isRoot) {}

  bool isChildConstant(const velox::common::ScanSpec& childSpec) const;

  /// Returns the operand of a struct. A struct below the root has no operand
  /// unless the scan reads it as a whole, which is not supported. Its fields
  /// are read as separate columns. A struct below an array or map is read into
  /// the elements of the array or map.
  static AbstractOperand* FOLLY_NULLABLE structOperand(
      const DefinesMap& defines,
      Path& path,
      bool isRoot,
      bool isNested);

 protected:
  /// Makes the reader of the field 'childSpec' of 'this' or of the elements,
  /// keys or values of an array or map. 'fileChildIdx' is the index of the
  /// child in the file type, -1 for a field that is not in the file.
  /// 'fileType' is the file type of the child or, for a field that is not in
  /// the file, made from 'requestedType'. 'path' ends with the child. A
  /// reader that adds its children otherwise need not implement this.
  virtual std::unique_ptr<ColumnReader> makeChild(
      velox::common::ScanSpec& /*childSpec*/,
      int32_t /*fileChildIdx*/,
      const TypePtr& /*requestedType*/,
      const std::shared_ptr<const dwio::common::TypeWithId>& /*fileType*/,
      Path& /*path*/,
      const DefinesMap& /*defines*/) {
    VELOX_UNREACHABLE();
  }

  /// Adds a child reader for each child of 'scanSpec' with makeChild(). A
  /// field that is in neither the requested type nor the file is an error. A
  /// field that is only filtered on is read with its file type. A field with a
  /// null constant in 'scanSpec' is made as not in the file. Call from the
  /// constructor of the most derived class.
  void addChildren(
      velox::common::ScanSpec& scanSpec,
      Path& path,
      const DefinesMap& defines);

  void addChild(std::unique_ptr<ColumnReader> child) {
    children_.push_back(child.get());
    childrenOwned_.push_back(std::move(child));
  }

  const bool isRoot_;
  std::vector<std::unique_ptr<ColumnReader>> childrenOwned_;
};

/// Reader of an array or map. The lengths of the top level rows or the rows of
/// the enclosing array give each row its range of elements. The children read
/// the elements of these ranges. A format makes the children in the
/// constructor of its subclass with addElements().
class RepeatedColumnReader : public StructColumnReader {
 public:
  RepeatedColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec,
      Path& path,
      const DefinesMap& defines)
      : StructColumnReader(
            requestedType,
            fileType,
            scalarOperand(defines, path, params),
            params,
            scanSpec,
            false) {}

  bool isRepeated() const override {
    return true;
  }

 protected:
  /// Adds the element reader of an array or the key and value readers of a
  /// map with makeChild(). Call from the constructor of the most derived
  /// class.
  void addElements(
      FormatParams& params,
      velox::common::ScanSpec& scanSpec,
      Path& path,
      const DefinesMap& defines);
};

} // namespace facebook::velox::wave
