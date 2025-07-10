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

#include "velox/connectors/hive/iceberg/Transforms.h"

#include "velox/connectors/hive/iceberg/Murmur3.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::hive::iceberg {

const TypePtr findChildTypeKind(
    const RowTypePtr& inputType,
    const std::string& fullName) {
  std::vector<std::string> parts;
  folly::split('.', fullName, parts);

  auto currentType = inputType->findChild(parts[0]);
  for (auto i = 1; i < parts.size(); ++i) {
    // The current type must be a ROW to continue traversal.
    VELOX_DCHECK_EQ(
        currentType->kind(),
        TypeKind::ROW,
        "Cannot access field '{}' in path '{}': '{}' is not a ROW type",
        parts[i],
        fullName,
        parts[i - 1]);
    const auto& rowType = asRowType(currentType);
    currentType = rowType->findChild(parts[i]);
  }
  return currentType;
}

// Iceberg spec requires URL encoding in the partition path.
std::string UrlEncode(const StringView& data) {
  std::string ret;
  ret.reserve(data.size() * 3);

  for (unsigned char c : data) {
    // These characters are not encoded in Java's URLEncoder.
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '*') {
      ret += c;
    } else if (c == ' ') {
      ret += '+';
    } else {
      // All other characters are percent-encoded.
      ret += fmt::format("%{:02X}", c);
    }
  }

  return ret;
}

template <typename T>
VectorPtr IdentityTransform<T>::apply(const VectorPtr& block) const {
  if constexpr (std::is_same_v<T, StringView>) {
    if (sourceType_->isVarchar()) {
      return block;
    }
  } else {
    return block;
  }

  auto result =
      BaseVector::create<FlatVector<T>>(sourceType_, block->size(), pool_);
  if (block->mayHaveNulls()) {
    result->setNulls(block->nulls());
  }

  DecodedVector decoded(*block);

  for (auto i = 0; i < block->size(); ++i) {
    if (!decoded.isNullAt(i)) {
      if constexpr (std::is_same_v<T, StringView>) {
        T value = decoded.valueAt<T>(i);
        std::string encodedValue =
            encoding::Base64::encode(value.data(), value.size());
        auto flatResult = result->template as<FlatVector<T>>();
        flatResult->set(i, StringView(encodedValue));
      }
    }
  }
  return result;
}

template <typename T>
VectorPtr BucketTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), block->size(), pool_);
  if (block->mayHaveNulls()) {
    result->setNulls(block->nulls());
  }
  DecodedVector decoded(*block);
  auto buckets = parameter_.value();
  for (auto i = 0; i < decoded.size(); ++i) {
    if (!decoded.isNullAt(i)) {
      T value = decoded.valueAt<T>(i);
      if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int128_t>) {
        if (sourceType_->isDecimal()) {
          result->set(i, Murmur3_32::hashDecimal(value) & 0x7FFFFFFF % buckets);
        } else {
          result->set(i, Murmur3_32::hash(value) & 0x7FFFFFFF % buckets);
        }
      } else {
        result->set(i, Murmur3_32::hash(value) & 0x7FFFFFFF % buckets);
      }
    }
  }
  return result;
}

template <typename T>
VectorPtr TruncateTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<T>>(sourceType_, block->size(), pool_);
  if (block->mayHaveNulls()) {
    result->setNulls(block->nulls());
  }

  DecodedVector decoded(*block);

  auto flatResult = result->template as<FlatVector<T>>();
  auto width = parameter_.value();
  char* rawBuffer = nullptr;

  if (std::is_same_v<T, StringView>) {
    if (sourceType_->isVarchar()) {
      rawBuffer =
          flatResult->getRawStringBufferWithSpace(block->size() * width);
    } else {
      rawBuffer = flatResult->getRawStringBufferWithSpace(
          block->size() * encoding::Base64::calculateEncodedSize(width));
    }
  }

  for (auto i = 0; i < block->size(); ++i) {
    if (!decoded.isNullAt(i)) {
      T value = decoded.valueAt<T>(i);
      if constexpr (
          std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
          std::is_same_v<T, int128_t>) {
        flatResult->set(i, value - ((value % width) + width) % width);
      } else if constexpr (std::is_same_v<T, StringView>) {
        if (sourceType_->isVarchar()) {
          auto length =
              functions::stringImpl::cappedByteLength<false>(value, width);
          if (StringView::isInline(length)) {
            flatResult->set(i, StringView(value.data(), length));
          } else {
            memcpy(rawBuffer, value.data(), length);
            flatResult->setNoCopy(i, StringView(rawBuffer, length));
            rawBuffer += length;
          }
        } else if (sourceType_->isVarbinary()) {
          std::string encoded = encoding::Base64::encode(
              value.data(), width > value.size() ? value.size() : width);
          auto length = encoded.length();
          if (StringView::isInline(length)) {
            flatResult->set(i, StringView(encoded));
          } else {
            memcpy(rawBuffer, encoded.data(), length);
            flatResult->setNoCopy(i, StringView(rawBuffer, length));
            rawBuffer += length;
          }
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

  if (block->mayHaveNulls()) {
    result->setNulls(block->nulls());
  }

  DecodedVector decoded(*block);

  auto flatResult = result->template as<FlatVector<int32_t>>();
  for (auto i = 0; i < block->size(); ++i) {
    if (!decoded.isNullAt(i)) {
      T value = decoded.valueAt<T>(i);
      flatResult->set(i, epochFunc_(value));
    }
  }

  return result;
}

template class IdentityTransform<bool>;
template class IdentityTransform<short>;
template class IdentityTransform<int32_t>;
template class IdentityTransform<int64_t>;
template class IdentityTransform<int128_t>;
template class IdentityTransform<Timestamp>;
template class IdentityTransform<signed char>;
template class IdentityTransform<double>;
template class IdentityTransform<float>;
template class IdentityTransform<StringView>;

template class BucketTransform<int32_t>;
template class BucketTransform<int64_t>;
template class BucketTransform<int128_t>;
template class BucketTransform<StringView>;
template class BucketTransform<Timestamp>;

template class TruncateTransform<int32_t>;
template class TruncateTransform<int64_t>;
template class TruncateTransform<int128_t>;
template class TruncateTransform<StringView>;

template class TemporalTransform<int32_t>;
template class TemporalTransform<int64_t>;
template class TemporalTransform<Timestamp>;

} // namespace facebook::velox::connector::hive::iceberg
