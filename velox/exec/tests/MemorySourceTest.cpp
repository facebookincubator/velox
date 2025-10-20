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

#include "velox/exec/MemorySource.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/MergeSource.h"
#include "velox/exec/SortBuffer.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/parse/PlanNodeIdGenerator.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

#include "utils/AssertQueryBuilder.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox;
using namespace facebook::velox::memory;

namespace facebook::velox::exec::test {

class MemorySourceTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
  }

  void produceAsync(
      RowVectorSource* source,
      const std::vector<RowVectorPtr>& vectors,
      size_t index = 0) const {
    ContinueFuture future{ContinueFuture::makeEmpty()};
    if (index >= vectors.size()) {
      const auto blockingReason = source->enqueue(nullptr, &future);
      EXPECT_EQ(blockingReason, BlockingReason::kNotBlocked);
      EXPECT_FALSE(future.valid());
      return;
    }

    const auto blockingReason = source->enqueue(vectors[index], &future);
    if (blockingReason == BlockingReason::kNotBlocked) {
      EXPECT_FALSE(future.valid());
      produceAsync(source, vectors, index + 1);
      return;
    }
    EXPECT_TRUE(future.valid());
    std::move(future)
        .via(executor_.get())
        .thenValue([this, source, &vectors, index](folly::Unit) {
          produceAsync(source, vectors, index + 1);
        })
        .thenError(folly::tag_t<std::exception>{}, [](const std::exception& e) {
          VELOX_FAIL(e.what());
        });
  }

  const RowTypePtr inputType_ =
      ROW({{"c0", BIGINT()}, {"c1", SMALLINT()}, {"c2", ROW({BIGINT()})}});
  const VectorFuzzer::Options fuzzerOpts_{.vectorSize = 5};
  const std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
};
} // namespace facebook::velox::exec::test

TEST_F(MemorySourceTest, basic) {
  for (auto i = 1; i < 16; ++i) {
    const auto source = RowVectorSource::createRowVectorSource(2);
    const auto vectors = createVectors(i, inputType_, fuzzerOpts_);
    std::thread thread([&]() { produceAsync(source.get(), vectors, 0); });
    const auto result =
        AssertQueryBuilder(
            PlanBuilder()
                .memorySource(
                    inputType_, reinterpret_cast<std::uintptr_t>(source.get()))
                .planNode())
            .assertResults(vectors);
    thread.join();
  }
}
