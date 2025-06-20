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

#include "velox/connectors/hive/iceberg/TransformFunction.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

int32_t offsetOfNthCodePoint(
    const char* const data,
    const size_t& len,
    const size_t& n) {
  int32_t offset{0};
  int32_t count{0}; // number of code points seen.

  while (offset < len) {
    if (count == n) {
      return offset;
    }
    const unsigned char c = data[offset];
    size_t charLen = 0;
    // 1-byte ASCII.
    if (c < 0x80) {
      charLen = 1;
    } else if ((c & 0xE0) == 0xC0) {
      charLen = 2;
    } else if ((c & 0xF0) == 0xE0) {
      charLen = 3;
    } else if ((c & 0xF8) == 0xF0) {
      charLen = 4;
    }
    offset += charLen;
    ++count;
  }
  return (count >= n) ? offset : len;
}

} // namespace

const TypePtr findChildTypeKind(
    const RowTypePtr& inputType,
    const std::string& path) {
  std::vector<std::string> parts;
  folly::split('.', path, parts);

  auto currentType = inputType->findChild(parts[0]);
  for (auto i = 1; i < parts.size(); ++i) {
    // The current type must be a ROW to continue traversal.
    if (currentType->kind() != TypeKind::ROW) {
      VELOX_USER_FAIL(
          "Cannot access field '{}' in path '{}': '{}' is not a ROW type",
          parts[i],
          path,
          parts[i - 1]);
    }

    const auto& rowType = currentType->as<TypeKind::ROW>();
    currentType = rowType.findChild(parts[i]);
  }
  return currentType;
}

template <typename T>
VectorPtr IdentityTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<T>>(sourceType_, block->size(), pool_);
  result->mutableNulls(block->size());

  DecodedVector decoded;
  SelectivityVector rows(block->size());
  decoded.decode(*block, rows);

  auto flatResult = result->template as<FlatVector<T>>();
  for (auto i = 0; i < block->size(); ++i) {
    if (decoded.isNullAt(i)) {
      result->setNull(i, true);
    } else {
      T value = decoded.valueAt<T>(i);
      if constexpr (std::is_same_v<T, StringView>) {
        std::string transformedValue(value);
        if (sourceType_->isVarbinary()) {
          transformedValue =
              encoding::Base64::encode(value.data(), value.size());
        }
        if (StringView::isInline(transformedValue.size())) {
          flatResult->set(i, StringView(transformedValue));
        } else {
          char* buffer =
              flatResult->getRawStringBufferWithSpace(transformedValue.size());
          memcpy(buffer, transformedValue.data(), transformedValue.size());
          auto sv = StringView(buffer, transformedValue.size());
          flatResult->set(i, sv);
        }
      } else {
        flatResult->set(i, value);
      }
    }
  }

  return std::static_pointer_cast<BaseVector>(result);
}

template <typename T>
VectorPtr BucketTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), block->size(), pool_);
  result->mutableNulls(block->size());
  auto flatResult = result->template as<FlatVector<int32_t>>();

  DecodedVector decoded;
  SelectivityVector rows(block->size());
  decoded.decode(*block, rows);

  for (auto i = 0; i < block->size(); ++i) {
    if (decoded.isNullAt(i)) {
      flatResult->setNull(i, true);
    } else {
      T value = decoded.valueAt<T>(i);
      uint32_t hashVal;
      if constexpr (std::is_same_v<T, int64_t>) {
        if (sourceType_->isShortDecimal()) {
          hashVal = connectors::hive::iceberg::Murmur3_32::hashDecimal(value);
        } else {
          hashVal = connectors::hive::iceberg::Murmur3_32::hash(value);
        }
      } else {
        hashVal = connectors::hive::iceberg::Murmur3_32::hash(value);
      }

      flatResult->set(i, (hashVal & 0x7FFFFFFF) % parameter_.value());
    }
  }

  return std::static_pointer_cast<BaseVector>(result);
}

template <typename T>
VectorPtr TruncateTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<T>>(sourceType_, block->size(), pool_);
  result->mutableNulls(block->size());

  DecodedVector decoded;
  SelectivityVector rows(block->size());
  decoded.decode(*block, rows);

  auto flatResult = result->template as<FlatVector<T>>();
  for (auto i = 0; i < block->size(); ++i) {
    if (decoded.isNullAt(i)) {
      result->setNull(i, true);
    } else {
      T value = decoded.valueAt<T>(i);
      const auto width = parameter_.value();
      if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
        flatResult->set(i, value - ((value % width) + width) % width);
      } else if constexpr (std::is_same_v<T, StringView>) {
        std::string transformedValue;
        if (sourceType_->isVarchar()) {
          const size_t offset =
              offsetOfNthCodePoint(value.data(), value.size(), width);
          transformedValue = std::string(value.data(), offset);
        } else if (sourceType_->isVarbinary()) {
          transformedValue = encoding::Base64::encode(
              value.data(), width > value.size() ? value.size() : width);
        }

        if (StringView::isInline(transformedValue.size())) {
          flatResult->set(i, StringView(transformedValue));
        } else {
          char* buffer =
              flatResult->getRawStringBufferWithSpace(transformedValue.size());
          memcpy(buffer, transformedValue.data(), transformedValue.size());
          auto sv = StringView(buffer, transformedValue.size());
          flatResult->set(i, sv);
        }
      }
    }
  }

  return result;
}

template <typename T>
VectorPtr TemporalTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), block->size(), pool_);
  result->mutableNulls(block->size());

  DecodedVector decoded;
  SelectivityVector rows(block->size());
  decoded.decode(*block, rows);

  auto flatResult = result->template as<FlatVector<int32_t>>();
  for (auto i = 0; i < block->size(); ++i) {
    if (decoded.isNullAt(i)) {
      result->setNull(i, true);
    } else {
      T value = decoded.valueAt<T>(i);
      flatResult->set(i, epochFunc_(value));
    }
  }

  return std::static_pointer_cast<BaseVector>(result);
}

template class IdentityTransform<bool>;
template class IdentityTransform<short>;
template class IdentityTransform<int32_t>;
template class IdentityTransform<int64_t>;
template class IdentityTransform<int128_t>;
template class IdentityTransform<StringView>;

template class BucketTransform<int32_t>;
template class BucketTransform<int64_t>;
template class BucketTransform<StringView>;

template class TruncateTransform<int32_t>;
template class TruncateTransform<int64_t>;
template class TruncateTransform<StringView>;

template class TemporalTransform<int32_t>;
template class TemporalTransform<int64_t>;

} // namespace facebook::velox::connector::hive::iceberg
