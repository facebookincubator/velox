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

#include "velox/common/base/Exceptions.h"
#include "velox/expression/EvalCtx.h"
#include "velox/functions/lib/MakeRowDefaultValue.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::functions {

///  A utility class for projecting specific keys from a vector of ROW type
///  Vector (which may be encoded) into a RowVector with named fields
///  corresponding to each projected key. For each active row, it extracts the
///  values corresponding to the specified keys and assigns them to their
///  respective fields in an output RowVector. The output RowVector contains one
///  field for each specified key, and its size will be equal to the last active
///  row in the selectivity vector.
///
///  Note:
/// - Rows marked as inactive in the 'rows' selectivity vector will result in
/// valid rows in the output; however, no guarantees are made regarding their
/// assigned values.
/// - Only hemogeneous row types are supported, e.g. ROW<INTEGER, BIGINT> as
/// input will fail.
class MakeRowFromStruct {
 public:
  struct KeyOptions {
    /// The key to extract from the row vector.
    std::string key;

    /// Given name of the output key. It is used to name the output field.
    std::string name;

    /// When set to true, if the key is missing or the value is null, the output
    /// is going to be padded with default values.
    bool padNull{false};
  };

  explicit MakeRowFromStruct(const std::vector<KeyOptions>& keysToProject)
      : outputFieldNames_{getFieldNames(keysToProject)},
        keysToProject_{keysToProject},
        padRequired_{isPaddingRequired(keysToProject)} {
    ensureValidFieldNames(keysToProject_);
  }

  VectorPtr apply(
      const BaseVector& rowVec,
      const SelectivityVector& rows,
      exec::EvalCtx* evalCtx) {
    VELOX_USER_CHECK_EQ(
        rowVec.type()->kind(), TypeKind::ROW, "Input must be of ROW typeKind");

    ensureValidInputType(rowVec.type()->asRow());

    return projectKeyImpl(rowVec, rows, evalCtx);
  }

 private:
  static void ensureValidFieldNames(
      const std::vector<KeyOptions>& keysToProject) {
    VELOX_USER_CHECK_GT(
        keysToProject.size(), 0, "Keys to project cannot be empty");

    std::unordered_set<std::string> fieldNamesSet;
    for (const auto& keyOption : keysToProject) {
      VELOX_USER_CHECK(!keyOption.name.empty(), "Field name cannot be empty");
      auto ok = fieldNamesSet.insert(keyOption.name).second;
      VELOX_USER_CHECK(
          ok, "Duplicate field names are not allowed: {}", keyOption.name);
    }
  }

  static std::vector<std::string> getFieldNames(
      const std::vector<KeyOptions>& keysToProject) {
    std::vector<std::string> fieldNames;
    fieldNames.reserve(keysToProject.size());
    for (const auto& keyOption : keysToProject) {
      fieldNames.push_back(keyOption.name);
    }
    return fieldNames;
  }

  static bool isPaddingRequired(const std::vector<KeyOptions>& keysToProject) {
    for (const auto& keyOption : keysToProject) {
      if (keyOption.padNull) {
        return true;
      }
    }
    return false;
  }

  static void ensureValidInputType(const RowType& rowType) {
    VELOX_USER_CHECK_GE(
        rowType.size(), 1, "Input row must have at least 1 field.");
    VELOX_USER_CHECK(
        rowType.size() == rowType.nameToIndex().size(),
        "Duplicate field names are not allowed: {}",
        rowType.toString());

    const auto& valueType = rowType.childAt(0);

    for (const auto& childType : rowType.children()) {
      VELOX_USER_CHECK_EQ(
          childType->kind(), valueType->kind(), "Child type mismatch");
    }
  }

  VectorPtr projectKeyImpl(
      const BaseVector& vector,
      const SelectivityVector& rows,
      exec::EvalCtx* evalCtx) {
    exec::LocalDecodedVector decodedRowVec(evalCtx);
    decodedRowVec.get()->decode(vector, rows);
    auto rowVec = decodedRowVec->base()->asUnchecked<RowVector>();
    const auto& rowType = rowVec->type()->asRow();
    const auto& valueType = rowType.childAt(0);
    auto outputSize = rows.end();

    std::vector<VectorPtr> children;
    children.reserve(keysToProject_.size());

    auto vectorPool = evalCtx ? evalCtx->vectorPool() : nullptr;
    for (size_t i = 0; i < keysToProject_.size(); ++i) {
      auto idx = rowType.getChildIdxIfExists(keysToProject_[i].key);

      // If the key does not need to be padded, simply copy the child vector or
      // create a null vector.
      if (!keysToProject_[i].padNull) {
        if (idx.has_value()) {
          children.push_back(rowVec->childAt(idx.value()));
        } else {
          children.push_back(
              BaseVector::createNullConstant(
                  valueType, outputSize, rowVec->pool()));
        }
        continue;
      }

      // Require padding.

      // If the key is missing, create a vector with default values.
      if (!idx.has_value()) {
        children.push_back(
            MakeRowDefaultValue::createFlat(
                valueType, outputSize, *rowVec->pool(), vectorPool));
        continue;
      }

      // If the key is present, copy the child vector if child does not contain
      // null. Otherwise, look into the child vector row by row and fill the
      // null values.
      auto& src = rowVec->childAt(idx.value());
      if (!src->mayHaveNulls()) {
        children.push_back(src);
      } else {
        exec::LocalDecodedVector decodedSrc(evalCtx);
        decodedSrc.get()->decode(*src, rows);
        auto srcBase = decodedSrc->base();

        auto dst = MakeRowDefaultValue::createFlat(
            valueType, outputSize, *rowVec->pool(), vectorPool);

        std::vector<BaseVector::CopyRange> ranges;
        rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
          if (!srcBase->isNullAt(row)) {
            ranges.emplace_back(row, row, 1);
          }
        });

        if (!ranges.empty()) {
          dst->copyRanges(srcBase, ranges);
        }
        children.push_back(dst);
      }
    }

    // If any padding is required, the final output vector will not have nulls.
    // Otherwise, respect the nulls masks from the input.
    auto outputNulls = padRequired_
        ? allocateNulls(outputSize, rowVec->pool(), bits::kNotNull)
        : rowVec->nulls();

    return std::make_shared<RowVector>(
        vector.pool(),
        ROW(outputFieldNames_, valueType),
        std::move(outputNulls),
        outputSize,
        std::move(children));
  }

  const std::vector<std::string> outputFieldNames_;
  const std::vector<KeyOptions> keysToProject_;
  const bool padRequired_{false};
};
} // namespace facebook::velox::functions
