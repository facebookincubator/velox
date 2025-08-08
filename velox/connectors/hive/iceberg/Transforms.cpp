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

namespace {

template <typename ProcessFunc>
FOLLY_ALWAYS_INLINE void transformValues(
    const VectorPtr& block,
    const DecodedVector* decoded,
    const VectorPtr& result,
    ProcessFunc&& processValue) {
  if (!decoded->mayHaveNulls()) {
    for (auto i = 0; i < decoded->size(); ++i) {
      processValue(i);
    }
  } else {
    block->mutableNulls(block->size());
    result->setNulls(block->nulls());
    for (auto i = 0; i < decoded->size(); ++i) {
      if (!decoded->isNullAt(i)) {
        processValue(i);
      }
    }
  }
}

} // namespace

VectorPtr Transform::transform(const RowVectorPtr& input) const {
  VectorPtr currentVector = input->childAt(sourceColumnName_);
  VELOX_CHECK_NOT_NULL(currentVector);
  return apply(currentVector);
}

std::string Transform::toHumanString(Timestamp value) const {
  TimestampToStringOptions options;
  options.precision = TimestampPrecision::kMilliseconds;
  options.zeroPaddingYear = true;
  options.skipTrailingZeros = true;
  options.leadingPositiveSign = true;
  options.skipTrailingZeroSeconds = true;
  return value.toString(options);
}

template <typename T>
VectorPtr IdentityTransform<T>::apply(const VectorPtr& block) const {
  if constexpr (!std::is_same_v<T, StringView>) {
    return block;
  }
  if (sourceType_->isVarchar()) {
    return block;
  }

  auto result =
      BaseVector::create<FlatVector<T>>(sourceType_, block->size(), pool_);
  DecodedVector decoded(*block);

  auto processValue = [&](auto i) {
    if constexpr (std::is_same_v<T, StringView>) {
      T value = decoded.valueAt<T>(i);
      auto encodedValue = encoding::Base64::encode(value.data(), value.size());
      result->set(i, StringView(encodedValue));
    }
  };

  transformValues(block, &decoded, result, processValue);
  return result;
}

template <typename T>
VectorPtr BucketTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), block->size(), pool_);

  DecodedVector decoded(*block);

  auto processValue = [&](auto i) {
    T value = decoded.valueAt<T>(i);
    int32_t hashValue;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int128_t>) {
      if (sourceType_->isDecimal()) {
        hashValue = Murmur3Hash32::hashDecimal(value);
      } else {
        hashValue = Murmur3Hash32::hash(value);
      }
    } else if constexpr (std::is_same_v<T, Timestamp>) {
      hashValue = Murmur3Hash32::hash(value.toMicros());
    } else {
      hashValue = Murmur3Hash32::hash(value);
    }
    result->set(i, (hashValue & 0x7FFFFFFF) % numBuckets_);
  };

  transformValues(block, &decoded, result, processValue);
  return result;
}

template <typename T>
VectorPtr TruncateTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<T>>(sourceType_, block->size(), pool_);

  auto flatResult = result->template as<FlatVector<T>>();
  char* rawBuffer = nullptr;
  BufferPtr buffer;
  if (std::is_same_v<T, StringView>) {
    if (sourceType_->isVarchar()) {
      buffer = result->getBufferWithSpace(block->size() * width_);
    } else {
      buffer = result->getBufferWithSpace(
          block->size() * encoding::Base64::calculateEncodedSize(width_));
    }
    rawBuffer = buffer->asMutable<char>() + buffer->size();
  }

  DecodedVector decoded(*block);
  auto processValue = [&](auto i) {
    T value = decoded.valueAt<T>(i);
    if constexpr (
        std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
        std::is_same_v<T, int128_t>) {
      flatResult->set(i, value - ((value % width_) + width_) % width_);
    } else if constexpr (std::is_same_v<T, StringView>) {
      if (sourceType_->isVarchar()) {
        auto length =
            functions::stringImpl::cappedByteLength<false>(value, width_);
        if (StringView::isInline(length)) {
          flatResult->set(i, StringView(value.data(), length));
        } else {
          memcpy(rawBuffer, value.data(), length);
          flatResult->setNoCopy(i, StringView(rawBuffer, length));
          rawBuffer += length;
        }
      } else if (sourceType_->isVarbinary()) {
        auto encoded = encoding::Base64::encode(
            value.data(), width_ > value.size() ? value.size() : width_);
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
  };

  transformValues(block, &decoded, result, processValue);

  if constexpr (std::is_same_v<T, StringView>) {
    buffer->setSize(rawBuffer - (buffer->asMutable<char>() + buffer->size()));
  }
  return result;
}

template <typename T>
VectorPtr TemporalTransform<T>::apply(const VectorPtr& block) const {
  auto result =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), block->size(), pool_);

  DecodedVector decoded(*block);
  auto processValue = [&](auto i) {
    T value = decoded.valueAt<T>(i);
    result->set(i, epochFunc_(value));
  };

  transformValues(block, &decoded, result, processValue);

  return result;
}

template class IdentityTransform<bool>;
template class IdentityTransform<int8_t>;
template class IdentityTransform<int16_t>;
template class IdentityTransform<int32_t>;
template class IdentityTransform<int64_t>;
template class IdentityTransform<int128_t>;
template class IdentityTransform<Timestamp>;
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
