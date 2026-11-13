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
#include <gtest/gtest.h>

#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/serializers/CompactRowSerializer.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/UnsafeRowSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::exec::test {

class RowContainerTestBase : public testing::Test,
                             public velox::test::VectorTestBase {
 protected:
  void SetUp() override {
    if (!isRegisteredVectorSerde()) {
      facebook::velox::serializer::presto::PrestoVectorSerde::
          registerVectorSerde();
    }
    if (!isRegisteredNamedVectorSerde("Presto")) {
      facebook::velox::serializer::presto::PrestoVectorSerde::
          registerNamedVectorSerde();
    }
    if (!isRegisteredNamedVectorSerde("CompactRow")) {
      facebook::velox::serializer::CompactRowVectorSerde::
          registerNamedVectorSerde();
    }
    if (!isRegisteredNamedVectorSerde("UnsafeRow")) {
      facebook::velox::serializer::spark::UnsafeRowVectorSerde::
          registerNamedVectorSerde();
    }
    filesystems::registerLocalFileSystem();
  }

  RowVectorPtr makeDataset(
      const TypePtr& rowType,
      const size_t size,
      std::function<void(RowVectorPtr)> customizeData) {
    auto batch = std::static_pointer_cast<RowVector>(
        velox::test::BatchMaker::createBatch(rowType, size, *pool_));
    if (customizeData) {
      customizeData(batch);
    }
    return batch;
  }
};
} // namespace facebook::velox::exec::test
