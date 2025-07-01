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

const TypePtr findChildTypeKind(
    const RowTypePtr& inputType,
    const std::string& fullName);

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

 private:
  const std::function<int32_t(T)> epochFunc_;
};

} // namespace facebook::velox::connector::hive::iceberg
