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

#include "Murmur3.h"

#include "velox/common/encode/Base64.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::hive::iceberg {

const TypePtr findChildTypeKind(
    const RowTypePtr& inputType,
    const std::string& path);

class Transform {
 public:
  Transform(
      TypePtr type,
      std::optional<int32_t> parameter,
      memory::MemoryPool* pool)
      : sourceType_(std::move(type)), parameter_(parameter), pool_(pool) {}

  virtual ~Transform() = default;

  virtual VectorPtr apply(const VectorPtr& block) const = 0;

  virtual const TypePtr resultType() const = 0;

  std::optional<int32_t> parameter() const {
    return parameter_;
  }

 protected:
  const TypePtr sourceType_;
  const std::optional<int32_t> parameter_;
  memory::MemoryPool* pool_;
};

template <typename T>
class IdentityTransform final : public Transform {
 public:
  IdentityTransform(const TypePtr& type, memory::MemoryPool* pool)
      : Transform(type, std::nullopt, pool) {}

  VectorPtr apply(const VectorPtr& block) const override;

  const TypePtr resultType() const override {
    return sourceType_;
  }
};

template <typename T>
class BucketTransform final : public Transform {
 public:
  BucketTransform(int32_t count, const TypePtr& type, memory::MemoryPool* pool)
      : Transform(type, count, pool) {}

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
      : Transform(type, width, pool) {}

  VectorPtr apply(const VectorPtr& block) const override;

  const TypePtr resultType() const override {
    return sourceType_;
  }
};

template <typename T>
class TemporalTransform final : public Transform {
 private:
  std::function<int32_t(int64_t)> epochFunc_;

 public:
  TemporalTransform(
      const std::function<int32_t(int64_t)>& epochFunc,
      const TypePtr& type,
      memory::MemoryPool* pool)
      : Transform(type, std::nullopt, pool), epochFunc_(epochFunc) {}

  VectorPtr apply(const VectorPtr& block) const override;

  const TypePtr resultType() const override {
    return INTEGER();
  }
};

} // namespace facebook::velox::connector::hive::iceberg
