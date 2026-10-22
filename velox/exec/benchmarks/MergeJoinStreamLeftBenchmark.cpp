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

// What `merge_join_stream_left_side` costs and what it buys.
//
// Streaming drops the left batches of an equal-key group once their rows have
// been emitted, instead of holding the whole group. What it buys is bounded
// left-side retention; what it costs is a reordering -- the right group has to
// be complete before any output -- paid on every group whether or not the
// group is wide enough for there to be anything to drop.
//
// This measures the cost side only, and the answer is that it is small. Across
// runs all three shapes land within noise of the buffered path, so time does
// not separate them and no shape shows a reliable win.
//
// The benefit cannot be measured here at all: a group is buffered by retaining
// shared_ptrs to input batches rather than copying rows, so the query pool's
// peak does not move either way. What streaming saves is retained input
// batches. Use this to confirm the config is cheap, not to decide it pays.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using facebook::velox::RowVector;
using facebook::velox::RowVectorPtr;
using facebook::velox::vector_size_t;
using facebook::velox::core::PlanNodeIdGenerator;
using facebook::velox::core::PlanNodePtr;
using facebook::velox::core::QueryConfig;
using facebook::velox::core::QueryCtx;
using facebook::velox::exec::test::AssertQueryBuilder;
using facebook::velox::exec::test::PlanBuilder;
using facebook::velox::memory::MemoryManager;

namespace {

// Rows per side. Same total work in every shape, so they differ only in how
// the rows are distributed across keys and batches.
constexpr vector_size_t kNumRows = 4'000'000;
// Rows per input batch. Groups have to outgrow this to span batches at all,
// which is the whole point of the wide case. Note this is close to the default
// preferred_output_batch_rows of 1024, so a left batch change lands roughly
// where the output vector would have been cut anyway.
constexpr vector_size_t kBatchSize = 1'000;
// A left batch far smaller than the output batch, so left batch changes no
// longer line up with output boundaries: about 16 of them per output vector,
// and a group spans thousands of batches rather than hundreds. That is where
// releasing consumed batches has the most to release.
constexpr vector_size_t kSmallLeftBatch = 64;
// A group far wider than a batch, and big enough that buffering it clears the
// pool's minimum reservation. Below that floor the saving is real but
// invisible.
constexpr vector_size_t kWideGroup = 1'000'000;
// Groups of a couple of rows, the shape measured on the query that regressed.
constexpr vector_size_t kNarrowGroup = 2;

class BenchmarkData : public facebook::velox::test::VectorTestBase {
 public:
  BenchmarkData(vector_size_t leftGroupSize, vector_size_t leftBatchSize) {
    // Ascending keys with 'leftGroupSize' rows each: merge join requires both
    // sides sorted on the key.
    auto leftKeys = makeFlatVector<int64_t>(
        kNumRows, [&](auto row) { return row / leftGroupSize; });
    auto leftPayload =
        makeFlatVector<int64_t>(kNumRows, [](auto row) { return row; });
    // Exactly one right row per distinct key. Repeating the left's grouping
    // here instead would make every key a group-by-group cross product, which
    // for the wide shape is 1M x 1M across each of the 4 keys and has nothing
    // to do with what is being measured.
    const vector_size_t numKeys = kNumRows / leftGroupSize;
    auto rightKeys =
        makeFlatVector<int64_t>(numKeys, [](auto row) { return row; });
    auto rightPayload =
        makeFlatVector<int64_t>(numKeys, [](auto row) { return row; });

    left_ = split(
        makeRowVector({"lk", "lv"}, {leftKeys, leftPayload}), leftBatchSize);
    right_ = split(
        makeRowVector({"rk", "rv"}, {rightKeys, rightPayload}), kBatchSize);

    plan_ = PlanBuilder(planIdGenerator_)
                .values(left_)
                .mergeJoin(
                    {"lk"},
                    {"rk"},
                    PlanBuilder(planIdGenerator_).values(right_).planNode(),
                    "",
                    {"lk", "lv", "rv"})
                .planNode();
  }

  void run(bool streamLeftSide) {
    auto queryCtx = QueryCtx::create(executor_.get());
    AssertQueryBuilder(plan_)
        .queryCtx(queryCtx)
        .config(
            QueryConfig::kMergeJoinStreamLeftSide,
            streamLeftSide ? "true" : "false")
        .copyResults(pool());
  }

 private:
  // Splits one vector into batches so a group can span more than one, which is
  // the only situation streaming can help with.
  std::vector<RowVectorPtr> split(
      const RowVectorPtr& data,
      vector_size_t batchSize) {
    std::vector<RowVectorPtr> batches;
    for (vector_size_t offset = 0; offset < data->size(); offset += batchSize) {
      const auto size =
          std::min<vector_size_t>(batchSize, data->size() - offset);
      batches.push_back(
          std::dynamic_pointer_cast<RowVector>(data->slice(offset, size)));
    }
    return batches;
  }

  std::shared_ptr<PlanNodeIdGenerator> planIdGenerator_{
      std::make_shared<PlanNodeIdGenerator>()};
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(4)};
  std::vector<RowVectorPtr> left_;
  std::vector<RowVectorPtr> right_;
  PlanNodePtr plan_;
};

// Held in function-local statics rather than globals so main() can destroy
// them before the MemoryManager goes away.
std::unique_ptr<BenchmarkData>& wideGroups() {
  static std::unique_ptr<BenchmarkData> data;
  return data;
}

std::unique_ptr<BenchmarkData>& narrowGroups() {
  static std::unique_ptr<BenchmarkData> data;
  return data;
}

std::unique_ptr<BenchmarkData>& smallLeftBatches() {
  static std::unique_ptr<BenchmarkData> data;
  return data;
}

} // namespace

// Wide left groups: a group spans many batches, so streaming has batches to
// drop. This is the shape the config exists for, though what it saves is
// retained batches rather than the time measured here.
BENCHMARK(wideGroupsBuffered) {
  wideGroups()->run(/*streamLeftSide=*/false);
}

BENCHMARK_RELATIVE(wideGroupsStreamed) {
  wideGroups()->run(/*streamLeftSide=*/true);
}

// Narrow left groups: a group fits in one batch, so there is nothing to drop
// and streaming has no benefit to offer at all. Included to price the
// reordering on its own, where it is the only thing being paid for.
BENCHMARK(narrowGroupsBuffered) {
  narrowGroups()->run(/*streamLeftSide=*/false);
}

BENCHMARK_RELATIVE(narrowGroupsStreamed) {
  narrowGroups()->run(/*streamLeftSide=*/true);
}

// Wide groups again, but split into left batches far smaller than the output
// batch, so a group spans ~15,600 of them instead of ~1,000. The other two
// shapes barely exercise the release at all by comparison; this is the one
// where dropping consumed batches has real work to do.
BENCHMARK(smallLeftBatchesBuffered) {
  smallLeftBatches()->run(/*streamLeftSide=*/false);
}

BENCHMARK_RELATIVE(smallLeftBatchesStreamed) {
  smallLeftBatches()->run(/*streamLeftSide=*/true);
}

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  MemoryManager::initialize(MemoryManager::Options{});

  wideGroups() = std::make_unique<BenchmarkData>(kWideGroup, kBatchSize);
  narrowGroups() = std::make_unique<BenchmarkData>(kNarrowGroup, kBatchSize);
  smallLeftBatches() =
      std::make_unique<BenchmarkData>(kWideGroup, kSmallLeftBatch);

  folly::runBenchmarks();

  wideGroups().reset();
  narrowGroups().reset();
  smallLeftBatches().reset();
  return 0;
}
