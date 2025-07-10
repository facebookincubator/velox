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

#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive::iceberg {

constexpr int32_t kEpochYear = 1970;

const TypePtr findChildTypeKind(
    const RowTypePtr& inputType,
    const std::string& fullName);

std::string UrlEncode(const StringView& data);

class Transform {
 public:
  Transform(
      TypePtr type,
      TransformType transformType,
      std::optional<int32_t> parameter,
      memory::MemoryPool* pool)
      : sourceType_(type),
        transformType_(transformType),
        parameter_(parameter),
        pool_(pool) {}

  virtual ~Transform() = default;

  virtual VectorPtr apply(const VectorPtr& block) const = 0;

  virtual const TypePtr resultType() const = 0;

  template <typename T>
  std::string toHumanString(T value) const {
    return folly::to<std::string>(value);
  }

  virtual std::string toHumanString(int32_t value) const {
    return folly::to<std::string>(value);
  }

  std::string toHumanString(int64_t value) const {
    if (sourceType_->isShortDecimal()) {
      return DecimalUtil::toString(value, sourceType_);
    }
    return folly::to<std::string>(value);
  }

  std::string toHumanString(int128_t value) const {
    if (sourceType_->isLongDecimal()) {
      return DecimalUtil::toString(value, sourceType_);
    }
    return folly::to<std::string>(value);
  }

  std::string toHumanString(const StringView& value) const {
    return UrlEncode(value);
  }

  std::string toHumanString(bool value) const {
    return value ? "true" : "false";
  }

  std::optional<int32_t> parameter() const {
    return parameter_;
  }

  std::string name() const {
    return transformTypeToName(transformType_);
  }

 protected:
  const TypePtr sourceType_;
  const TransformType transformType_;
  const std::optional<int32_t> parameter_;
  memory::MemoryPool* pool_;
};

template <typename T>
class IdentityTransform final : public Transform {
 public:
  IdentityTransform(const TypePtr& type, memory::MemoryPool* pool)
      : Transform(type, TransformType::kIdentity, std::nullopt, pool) {}

  VectorPtr apply(const VectorPtr& block) const override;

  const TypePtr resultType() const override {
    return sourceType_;
  }

  std::string toHumanString(int32_t value) const override {
    if (sourceType_->isDate()) {
      return DATE()->toString(value);
    }
    return folly::to<std::string>(value);
  }
};

template <typename T>
class BucketTransform final : public Transform {
 public:
  BucketTransform(int32_t count, const TypePtr& type, memory::MemoryPool* pool)
      : Transform(type, TransformType::kBucket, count, pool) {}

  VectorPtr apply(const VectorPtr& block) const override;

  const TypePtr resultType() const override {
    return INTEGER();
  }
};

template <typename T>
class TruncateTransform final : public Transform {
 public:
  TruncateTransform(
      int32_t width,
      const TypePtr& type,
      memory::MemoryPool* pool)
      : Transform(type, TransformType::kTruncate, width, pool) {}

  VectorPtr apply(const VectorPtr& block) const override;

  const TypePtr resultType() const override {
    return sourceType_;
  }
};

template <typename T>
class TemporalTransform final : public Transform {
 public:
  TemporalTransform(
      const TypePtr& type,
      TransformType transformType,
      memory::MemoryPool* pool,
      const std::function<int32_t(T)>& epochFunc)
      : Transform(type, transformType, std::nullopt, pool),
        epochFunc_(epochFunc) {}

  VectorPtr apply(const VectorPtr& block) const override;

  const TypePtr resultType() const override {
    return INTEGER();
  }

  std::string toHumanString(int32_t value) const override {
    switch (transformType_) {
      case TransformType::kYear: {
        return fmt::format("{:04d}", kEpochYear + value);
      }
      case TransformType::kMonth: {
        int32_t year = kEpochYear + value / 12;
        int32_t month = 1 + value % 12;
        if (month <= 0) {
          month += 12;
          year -= 1;
        }
        return fmt::format("{:04d}-{:02d}", year, month);
      }
      case TransformType::kHour: {
        int64_t seconds = static_cast<int64_t>(value) * 3600;
        std::tm tmValue;
        VELOX_USER_CHECK(
            Timestamp::epochToCalendarUtc(seconds, tmValue),
            "Can't convert seconds {}*3600 to time.",
            seconds);

        return fmt::format(
            "{:04d}-{:02d}-{:02d}-{:02d}",
            tmValue.tm_year + 1900,
            tmValue.tm_mon + 1,
            tmValue.tm_mday,
            tmValue.tm_hour);
      }
      case TransformType::kDay:
      default: {
        return DATE()->toString(value);
      }
    }
  }

 private:
  const std::function<int32_t(T)> epochFunc_;
};

} // namespace facebook::velox::connector::hive::iceberg
