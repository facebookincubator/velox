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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

/// Benchmark for SpatialJoin operator, which implements a nested-loop join
/// with spatial predicates (e.g., ST_INTERSECTS, ST_CONTAINS, ST_WITHIN).
///
/// This benchmark measures the performance of spatial joins under different
/// conditions:
/// - Different build and probe side sizes (cross join cardinality)
/// - Different spatial predicates
/// - Different data distributions (dense vs sparse geometries)
/// - Inner vs Left join types
///
/// The benchmark creates synthetic geometric data and measures the throughput
/// of spatial join operations. The focus is on understanding how the nested
/// loop pattern performs with varying data sizes and selectivity.

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

/// Spatial distribution patterns for geometry generation.
enum class Distribution {
  kUniform, // Geometries uniformly distributed in space
  kClustered // Geometries clustered in specific regions
};

// Constants for geometry generation.
constexpr int32_t kNullPatternModulo = 13;
constexpr int32_t kRandomCoordinateMax = 10000;
constexpr double kCoordinateScaleDivisor = 10.0;
constexpr int32_t kNumClusters = 5;
constexpr double kClusterSpacing = 200.0;
constexpr double kClusterCenterOffset = 100.0;
constexpr int32_t kClusterSpreadRange = 100;
constexpr int32_t kClusterSpreadHalf = 50;
constexpr double kPolygonSize = 10.0;

// Constants for benchmark configuration.
constexpr int32_t kDefaultBatchSize = 10000;
constexpr int32_t kSmallBenchmarkSize = 1000;
constexpr int32_t kMediumProbeBenchmarkSize = 50000;
constexpr int32_t kMediumBuildBenchmarkSize = 5000;
constexpr int32_t kLargeProbeBenchmarkSize = 200000;
constexpr int32_t kLargeBuildBenchmarkSize = 50000;

/// Parameters for a spatial join benchmark test case.
struct SpatialJoinBenchmarkParams {
  /// Number of rows on the probe (left) side.
  int32_t probeSize;

  /// Number of rows on the build (right) side.
  int32_t buildSize;

  /// Spatial predicate to use (e.g., "ST_Intersects", "ST_Contains").
  std::string predicate;

  /// Join type (kInner or kLeft).
  core::JoinType joinType;

  /// Spatial distribution pattern for geometry generation.
  Distribution distribution;

  /// Description for benchmark naming.
  std::string toString() const {
    std::string joinTypeStr =
        (joinType == core::JoinType::kInner) ? "Inner" : "Left";
    std::string distributionStr =
        (distribution == Distribution::kUniform) ? "uniform" : "clustered";
    return fmt::format(
        "{}x{}_{}_{}_{}",
        probeSize,
        buildSize,
        predicate,
        joinTypeStr,
        distributionStr);
  }
};

class SpatialJoinBenchmark : public facebook::velox::test::VectorTestBase {
 public:
  SpatialJoinBenchmark() : rng_((std::random_device{}())) {}

  /// Creates a vector of POINT geometries with specified distribution.
  VectorPtr
  makePointVector(int32_t size, Distribution distribution, bool nulls = false) {
    return makeFlatVector<std::string>(
        size,
        [&](vector_size_t row) {
          if (nulls && (row % kNullPatternModulo == 0)) {
            return std::string("");
          }
          double x, y;
          if (distribution == Distribution::kUniform) {
            x = (folly::Random::rand32(rng_) % kRandomCoordinateMax) /
                kCoordinateScaleDivisor;
            y = (folly::Random::rand32(rng_) % kRandomCoordinateMax) /
                kCoordinateScaleDivisor;
          } else {
            int cluster = row % kNumClusters;
            double centerX = (cluster * kClusterSpacing) + kClusterCenterOffset;
            double centerY = (cluster * kClusterSpacing) + kClusterCenterOffset;
            x = centerX +
                ((folly::Random::rand32(rng_) % kClusterSpreadRange) -
                 kClusterSpreadHalf);
            y = centerY +
                ((folly::Random::rand32(rng_) % kClusterSpreadRange) -
                 kClusterSpreadHalf);
          }
          return fmt::format("POINT ({} {})", x, y);
        },
        [&](vector_size_t row) {
          return nulls && (row % kNullPatternModulo == 0);
        });
  }

  /// Creates a vector of POLYGON geometries with specified distribution.
  VectorPtr makePolygonVector(
      int32_t size,
      Distribution distribution,
      bool nulls = false) {
    return makeFlatVector<std::string>(
        size,
        [&](vector_size_t row) {
          if (nulls && (row % kNullPatternModulo == 0)) {
            return std::string("");
          }
          double centerX, centerY;
          if (distribution == Distribution::kUniform) {
            centerX = (folly::Random::rand32(rng_) % kRandomCoordinateMax) /
                kCoordinateScaleDivisor;
            centerY = (folly::Random::rand32(rng_) % kRandomCoordinateMax) /
                kCoordinateScaleDivisor;
          } else {
            int cluster = row % kNumClusters;
            centerX = (cluster * kClusterSpacing) + kClusterCenterOffset;
            centerY = (cluster * kClusterSpacing) + kClusterCenterOffset;
          }
          return fmt::format(
              "POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))",
              centerX - kPolygonSize,
              centerY - kPolygonSize,
              centerX + kPolygonSize,
              centerY - kPolygonSize,
              centerX + kPolygonSize,
              centerY + kPolygonSize,
              centerX - kPolygonSize,
              centerY + kPolygonSize,
              centerX - kPolygonSize,
              centerY - kPolygonSize);
        },
        [&](vector_size_t row) {
          return nulls && (row % kNullPatternModulo == 0);
        });
  }

  RowVectorPtr createProjectionVector(
      const std::string& prefix,
      RowVectorPtr input) {
    const auto plan = PlanBuilder(std::make_shared<core::PlanNodeIdGenerator>())
                          .values({input})
                          .project(
                              {fmt::format("{}_id", prefix),
                               fmt::format(
                                   "ST_GeometryFromText({}_geom) AS {}_geom",
                                   prefix,
                                   prefix)})
                          .planNode();
    return AssertQueryBuilder(plan).copyResults(pool_.get());
  }

  /// Creates test data for the specified parameters.
  std::pair<std::vector<RowVectorPtr>, std::vector<RowVectorPtr>> makeTestData(
      const SpatialJoinBenchmarkParams& params) {
    // Create probe side data (points)
    std::vector<RowVectorPtr> probeVectors;
    const int32_t batchSize = std::min(params.probeSize, kDefaultBatchSize);
    const int32_t numBatches = (params.probeSize + batchSize - 1) / batchSize;

    for (int32_t i = 0; i < numBatches; ++i) {
      int32_t currentBatchSize =
          std::min(batchSize, params.probeSize - (i * batchSize));
      auto geomVector =
          makePointVector(currentBatchSize, params.distribution, false);
      auto idVector = makeFlatVector<int64_t>(
          currentBatchSize,
          [i, batchSize](vector_size_t row) { return (i * batchSize) + row; });
      probeVectors.push_back(createProjectionVector(
          "probe",
          makeRowVector({"probe_id", "probe_geom"}, {idVector, geomVector})));
    }

    // Create build side data (polygons)
    std::vector<RowVectorPtr> buildVectors;
    const int32_t buildBatchSize =
        std::min(params.buildSize, kDefaultBatchSize);
    const int32_t numBuildBatches =
        (params.buildSize + buildBatchSize - 1) / buildBatchSize;

    for (int32_t i = 0; i < numBuildBatches; ++i) {
      int32_t currentBatchSize =
          std::min(buildBatchSize, params.buildSize - (i * buildBatchSize));
      auto geomVector =
          makePolygonVector(currentBatchSize, params.distribution, false);
      auto idVector = makeFlatVector<int64_t>(
          currentBatchSize, [i, buildBatchSize](vector_size_t row) {
            return (i * buildBatchSize) + row;
          });
      buildVectors.push_back(createProjectionVector(
          "build",
          makeRowVector({"build_id", "build_geom"}, {idVector, geomVector})));
    }

    return {probeVectors, buildVectors};
  }

  /// Creates a spatial join plan with the specified parameters.
  std::shared_ptr<const core::PlanNode> makeSpatialJoinPlan(
      std::vector<RowVectorPtr>&& probeVectors,
      std::vector<RowVectorPtr>&& buildVectors,
      const SpatialJoinBenchmarkParams& params) {
    const auto planNodeIdGenerator =
        std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values(probeVectors)
        .spatialJoin(
            PlanBuilder(planNodeIdGenerator).values(buildVectors).planNode(),
            fmt::format("{}(probe_geom, build_geom)", params.predicate),
            "probe_geom",
            "build_geom",
            std::nullopt,
            {"probe_id", "probe_geom", "build_id", "build_geom"},
            params.joinType)
        .planNode();
  }

  /// Runs a single benchmark iteration.
  uint64_t run(
      std::shared_ptr<const core::PlanNode> plan,
      const SpatialJoinBenchmarkParams& params) {
    auto result = AssertQueryBuilder(plan).copyResults(pool_.get());
    return result->size();
  }

  /// Adds a benchmark for the given parameters.
  void addBenchmark(const SpatialJoinBenchmarkParams& params) {
    auto name = params.toString();
    folly::addBenchmark(__FILE__, name, [this, params]() {
      std::shared_ptr<const core::PlanNode> plan;
      BENCHMARK_SUSPEND {
        auto [probeVectors, buildVectors] = makeTestData(params);
        plan = makeSpatialJoinPlan(
            std::move(probeVectors), std::move(buildVectors), params);
      }

      run(plan, params);
      return 1;
    });
  }

 private:
  std::default_random_engine rng_;
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::initializeMemoryManager(memory::MemoryManager::Options{});
  parse::registerTypeResolver();
  functions::prestosql::registerAllScalarFunctions();

  SpatialJoinBenchmark bm;

  // Small scale benchmarks (1K x 1K)
  bm.addBenchmark(
      {kSmallBenchmarkSize,
       kSmallBenchmarkSize,
       "ST_Intersects",
       core::JoinType::kInner,
       Distribution::kUniform});
  bm.addBenchmark(
      {kSmallBenchmarkSize,
       kSmallBenchmarkSize,
       "ST_Intersects",
       core::JoinType::kInner,
       Distribution::kClustered});

  // Medium scale benchmarks (50K x 5K)
  bm.addBenchmark(
      {kMediumProbeBenchmarkSize,
       kMediumBuildBenchmarkSize,
       "ST_Intersects",
       core::JoinType::kInner,
       Distribution::kUniform});
  bm.addBenchmark(
      {kMediumProbeBenchmarkSize,
       kMediumBuildBenchmarkSize,
       "ST_Intersects",
       core::JoinType::kInner,
       Distribution::kClustered});

  // Left join benchmarks (50K x 5K)
  bm.addBenchmark(
      {kMediumProbeBenchmarkSize / 2,
       kMediumBuildBenchmarkSize,
       "ST_Intersects",
       core::JoinType::kLeft,
       Distribution::kUniform});
  bm.addBenchmark(
      {kMediumProbeBenchmarkSize / 2,
       kMediumBuildBenchmarkSize,
       "ST_Intersects",
       core::JoinType::kLeft,
       Distribution::kClustered});

  // Contains predicate benchmarks (50K x 5K)
  bm.addBenchmark(
      {kMediumProbeBenchmarkSize / 2,
       kMediumBuildBenchmarkSize,
       "ST_Contains",
       core::JoinType::kInner,
       Distribution::kUniform});

  // Large scale benchmark (200K x 50K)
  bm.addBenchmark(
      {kLargeProbeBenchmarkSize,
       kLargeBuildBenchmarkSize,
       "ST_Intersects",
       core::JoinType::kInner,
       Distribution::kUniform});

  folly::runBenchmarks();
  return 0;
}
