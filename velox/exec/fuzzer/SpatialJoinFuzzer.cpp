/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/fuzzer/SpatialJoinFuzzer.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/fuzzer/Utils.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int32(steps, 10, "Number of plans to generate and test.");

DEFINE_int32(
    duration_sec,
    0,
    "For how long it should run (in seconds). If zero, "
    "it executes exactly --steps iterations and exits.");

DEFINE_int32(
    batch_size,
    100,
    "The number of elements on each generated vector.");

DEFINE_int32(num_batches, 10, "The number of generated vectors.");

DEFINE_double(
    null_ratio,
    0.1,
    "Chance of adding a null value in a vector "
    "(expressed as double from 0 to 1).");

namespace facebook::velox::exec {

namespace {
using namespace facebook::velox;

/// Spatial distribution patterns for geometry generation.
enum class GeometryDistribution {
  kUniform, // Geometries uniformly distributed in space
  kClustered, // Geometries clustered in specific regions
  kSparse // Sparse geometries with low overlap probability
};

// Constants for geometry generation.
constexpr int32_t kRandomCoordinateMax = 1000;
constexpr int32_t kNumClusters = 5;
constexpr double kClusterSpacing = 200.0;
constexpr double kClusterCenterOffset = 100.0;
constexpr int32_t kClusterSpreadRange = 100;
constexpr int32_t kClusterSpreadHalf = kClusterSpreadRange / 2;
constexpr double kPolygonSize = 10.0;
constexpr double kSparseSpread = 2000.0;
constexpr uint32_t kMaxRadius = 100;

// Base class for geometry string generators.
class GeometryInputGenerator : public AbstractInputGenerator {
 public:
  GeometryInputGenerator(
      GeometryDistribution distribution,
      size_t seed,
      double nullRatio)
      : AbstractInputGenerator(seed, VARCHAR(), nullptr, nullRatio),
        distribution_(distribution) {}

 protected:
  std::pair<double, double> generateCoordinates() {
    double x, y;
    switch (distribution_) {
      case GeometryDistribution::kUniform: {
        x = fuzzer::rand<int32_t>(
            rng_, -kRandomCoordinateMax, kRandomCoordinateMax);
        y = fuzzer::rand<int32_t>(
            rng_, -kRandomCoordinateMax, kRandomCoordinateMax);
        break;
      }
      case GeometryDistribution::kClustered: {
        uint32_t cluster = fuzzer::rand<uint32_t>(rng_, 0, kNumClusters);
        double centerX = (cluster * kClusterSpacing) + kClusterCenterOffset;
        double centerY = (cluster * kClusterSpacing) + kClusterCenterOffset;
        x = centerX +
            ((fuzzer::rand<int32_t>(
                 rng_, -kClusterSpreadRange, kClusterSpreadRange)) -
             kClusterSpreadHalf);
        y = centerY +
            ((fuzzer::rand<int32_t>(
                 rng_, -kClusterSpreadRange, kClusterSpreadRange)) -
             kClusterSpreadHalf);
        break;
      }
      case GeometryDistribution::kSparse: {
        x = fuzzer::rand<int32_t>(rng_, -kSparseSpread, kSparseSpread);
        y = fuzzer::rand<int32_t>(rng_, -kSparseSpread, kSparseSpread);
        break;
      }
    }
    return {x, y};
  }

  GeometryDistribution distribution_;
};

// Generates POINT geometry strings.
class PointInputGenerator : public GeometryInputGenerator {
 public:
  PointInputGenerator(
      GeometryDistribution distribution,
      size_t seed,
      double nullRatio)
      : GeometryInputGenerator(distribution, seed, nullRatio) {}

  variant generate() override {
    if (fuzzer::coinToss(rng_, nullRatio_)) {
      return variant::null(type_->kind());
    }

    auto [x, y] = generateCoordinates();
    return fmt::format("POINT ({} {})", x, y);
  }
};

// Generates POLYGON geometry strings.
class PolygonInputGenerator : public GeometryInputGenerator {
 public:
  PolygonInputGenerator(
      GeometryDistribution distribution,
      size_t seed,
      double nullRatio)
      : GeometryInputGenerator(distribution, seed, nullRatio) {}

  variant generate() override {
    if (fuzzer::coinToss(rng_, nullRatio_)) {
      return variant::null(type_->kind());
    }
    auto [centerX, centerY] = generateCoordinates();
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
  }
};

// Generates LINESTRING geometry strings.
class LineStringInputGenerator : public GeometryInputGenerator {
 public:
  LineStringInputGenerator(
      GeometryDistribution distribution,
      size_t seed,
      double nullRatio)
      : GeometryInputGenerator(distribution, seed, nullRatio) {}

  variant generate() override {
    if (fuzzer::coinToss(rng_, nullRatio_)) {
      return variant::null(type_->kind());
    }
    auto [x1, y1] = generateCoordinates();
    double x2 = x1 + kPolygonSize;
    double y2 = y1 + kPolygonSize;
    return fmt::format("LINESTRING ({} {}, {} {})", x1, y1, x2, y2);
  }
};

class SpatialJoinFuzzer {
 public:
  explicit SpatialJoinFuzzer(size_t initialSeed);

  void go();

 private:
  static VectorFuzzer::Options getFuzzerOptions() {
    VectorFuzzer::Options opts;
    opts.vectorSize = FLAGS_batch_size;
    opts.stringVariableLength = true;
    opts.stringLength = 100;
    opts.nullRatio = FLAGS_null_ratio;
    return opts;
  }

  void seed(size_t seed) {
    currentSeed_ = seed;
    vectorFuzzer_.reSeed(seed);
    rng_.seed(currentSeed_);
  }

  void reSeed() {
    seed(rng_());
  }

  // Randomly pick a join type supported by SpatialJoin.
  core::JoinType pickJoinType();

  // Randomly pick a spatial predicate function.
  std::string pickSpatialPredicate();

  // Randomly pick a geometry distribution pattern.
  GeometryDistribution pickDistribution();

  // Runs one test iteration from query plans generation, execution and result
  // verification.
  void verify(core::JoinType joinType);

  // Creates a vector of POINT geometries with specified distribution.
  VectorPtr makePointVector(int32_t size, GeometryDistribution distribution);

  // Creates a vector of POLYGON geometries with specified distribution.
  VectorPtr makePolygonVector(int32_t size, GeometryDistribution distribution);

  // Creates a vector of LINESTRING geometries with specified distribution.
  VectorPtr makeLineStringVector(
      int32_t size,
      GeometryDistribution distribution);

  // Returns randomly generated probe input with geometry columns (as WKT
  // strings).
  std::vector<RowVectorPtr> generateProbeInput(
      GeometryDistribution distribution);

  // Same as generateProbeInput() but copies over 10% of the input to ensure
  // some matches during joining. Also generates an empty input with a 10%
  // chance.
  std::vector<RowVectorPtr> generateBuildInput(
      const std::vector<RowVectorPtr>& probeInput,
      GeometryDistribution distribution);

  // Executes a plan and returns the result.
  RowVectorPtr execute(const core::PlanNodePtr& plan);

  int32_t randInt(int32_t min, int32_t max) {
    return boost::random::uniform_int_distribution<int32_t>(min, max)(rng_);
  }

  const std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  std::mt19937 rng_;
  size_t currentSeed_{0};

  VectorFuzzer vectorFuzzer_;

  struct {
    size_t numIterations{0};
  } stats_;
};

SpatialJoinFuzzer::SpatialJoinFuzzer(size_t initialSeed)
    : vectorFuzzer_{getFuzzerOptions(), pool_.get()} {
  filesystems::registerLocalFileSystem();
  seed(initialSeed);
}

template <typename T>
bool isDone(size_t i, T startTime) {
  if (FLAGS_duration_sec > 0) {
    std::chrono::duration<double> elapsed =
        std::chrono::system_clock::now() - startTime;
    return elapsed.count() >= FLAGS_duration_sec;
  }
  return i >= FLAGS_steps;
}

core::JoinType SpatialJoinFuzzer::pickJoinType() {
  // SpatialJoin only supports INNER and LEFT join types.
  static std::vector<core::JoinType> kJoinTypes = {
      core::JoinType::kInner, core::JoinType::kLeft};

  const size_t idx = randInt(0, kJoinTypes.size() - 1);
  return kJoinTypes[idx];
}

std::string SpatialJoinFuzzer::pickSpatialPredicate() {
  // Common spatial predicates supported by spatial joins.
  static std::vector<std::string> kPredicates = {
      "ST_Intersects",
      "ST_Contains",
      "ST_Within",
      "ST_Distance",
      "ST_Overlaps",
      "ST_Crosses",
      "ST_Touches",
      "ST_Equals"};

  const size_t idx = randInt(0, kPredicates.size() - 1);
  return kPredicates[idx];
}

GeometryDistribution SpatialJoinFuzzer::pickDistribution() {
  static std::vector<GeometryDistribution> kDistributions = {
      GeometryDistribution::kUniform,
      GeometryDistribution::kClustered,
      GeometryDistribution::kSparse};

  const size_t idx = randInt(0, kDistributions.size() - 1);
  return kDistributions[idx];
}

VectorPtr SpatialJoinFuzzer::makePointVector(
    int32_t size,
    GeometryDistribution distribution) {
  auto generator = std::make_shared<PointInputGenerator>(
      distribution, currentSeed_, getFuzzerOptions().nullRatio);
  return vectorFuzzer_.fuzzFlat(VARCHAR(), size, generator);
}

VectorPtr SpatialJoinFuzzer::makePolygonVector(
    int32_t size,
    GeometryDistribution distribution) {
  auto generator = std::make_shared<PolygonInputGenerator>(
      distribution, currentSeed_, getFuzzerOptions().nullRatio);
  return vectorFuzzer_.fuzzFlat(VARCHAR(), size, generator);
}

VectorPtr SpatialJoinFuzzer::makeLineStringVector(
    int32_t size,
    GeometryDistribution distribution) {
  auto generator = std::make_shared<LineStringInputGenerator>(
      distribution, currentSeed_, getFuzzerOptions().nullRatio);
  return vectorFuzzer_.fuzzFlat(VARCHAR(), size, generator);
}

std::vector<RowVectorPtr> SpatialJoinFuzzer::generateProbeInput(
    GeometryDistribution distribution) {
  std::vector<RowVectorPtr> input;

  const int32_t numRows = FLAGS_batch_size * FLAGS_num_batches;
  const int32_t batchSize = FLAGS_batch_size;
  const int32_t numBatches = FLAGS_num_batches;

  // Randomly pick geometry type for probe side.
  const int geometryType = randInt(0, 2);

  for (int32_t i = 0; i < numBatches; ++i) {
    int32_t currentBatchSize = std::min(batchSize, numRows - (i * batchSize));

    VectorPtr geomVector;
    if (geometryType == 0) {
      geomVector = makePointVector(currentBatchSize, distribution);
    } else if (geometryType == 1) {
      geomVector = makePolygonVector(currentBatchSize, distribution);
    } else {
      geomVector = makeLineStringVector(currentBatchSize, distribution);
    }

    auto idVector = vectorFuzzer_.fuzzFlat(BIGINT(), currentBatchSize);
    auto rowType = ROW(
        {"probe_id", "probe_geom_wkt"}, {idVector->type(), geomVector->type()});
    auto rowVector = std::make_shared<RowVector>(
        pool_.get(),
        rowType,
        nullptr,
        currentBatchSize,
        std::vector{idVector, geomVector});
    input.push_back(rowVector);
  }

  return input;
}

std::vector<RowVectorPtr> SpatialJoinFuzzer::generateBuildInput(
    const std::vector<RowVectorPtr>& probeInput,
    GeometryDistribution distribution) {
  std::vector<RowVectorPtr> input;

  // 1 in 10 times use empty build.
  if (vectorFuzzer_.coinToss(0.1)) {
    auto rowType = ROW({"build_id", "build_geom_wkt"}, {BIGINT(), VARCHAR()});
    auto rowVector = std::make_shared<RowVector>(
        pool_.get(),
        rowType,
        nullptr,
        0,
        std::vector{
            vectorFuzzer_.fuzzFlat(BIGINT(), 0),
            vectorFuzzer_.fuzzFlat(VARCHAR(), 0)});
    return {rowVector};
  }

  // Randomly pick geometry type for build side.
  const int geometryType = randInt(0, 2);

  for (const auto& probe : probeInput) {
    auto numRows = 1 + probe->size() / 8;

    VectorPtr geomVector;
    if (geometryType == 0) {
      geomVector = makePointVector(numRows, distribution);
    } else if (geometryType == 1) {
      geomVector = makePolygonVector(numRows, distribution);
    } else {
      geomVector = makeLineStringVector(numRows, distribution);
    }

    auto idVector = vectorFuzzer_.fuzzFlat(BIGINT(), numRows);

    // To ensure some matches, copy some geometries from probe side.
    if (probe->size() > 0) {
      std::vector<vector_size_t> rowNumbers(numRows);
      SelectivityVector rows(numRows, false);
      for (vector_size_t i = 0; i < numRows; ++i) {
        if (vectorFuzzer_.coinToss(0.3)) {
          rowNumbers[i] = randInt(0, probe->size() - 1);
          rows.setValid(i, true);
        }
      }

      // Copy geometry from probe to build.
      auto probeGeom = probe->childAt(1);
      geomVector->copy(probeGeom.get(), rows, rowNumbers.data());
    }

    auto rowType = ROW(
        {"build_id", "build_geom_wkt"}, {idVector->type(), geomVector->type()});
    auto rowVector = std::make_shared<RowVector>(
        pool_.get(),
        rowType,
        nullptr,
        numRows,
        std::vector{idVector, geomVector});
    input.push_back(rowVector);
  }

  return input;
}

RowVectorPtr SpatialJoinFuzzer::execute(const core::PlanNodePtr& plan) {
  LOG(INFO) << "Executing query plan: " << std::endl
            << plan->toString(true, true);

  return test::AssertQueryBuilder(plan).copyResults(pool_.get());
}

void SpatialJoinFuzzer::verify(core::JoinType joinType) {
  const auto distribution = pickDistribution();
  const auto predicate = pickSpatialPredicate();

  // Generate test data (WKT strings).
  auto probeInput = generateProbeInput(distribution);
  auto buildInput = generateBuildInput(probeInput, distribution);

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Probe input: " << probeInput[0]->toString();
    for (const auto& v : probeInput) {
      VLOG(1) << std::endl << v->toString(0, v->size());
    }

    VLOG(1) << "Build input: " << buildInput[0]->toString();
    for (const auto& v : buildInput) {
      VLOG(1) << std::endl << v->toString(0, v->size());
    }
  }

  // Build spatial join plan with geometry conversion as part of the plan.
  const auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();

  std::string joinCondition;
  std::optional<std::string> radiusColumn;
  std::optional<std::string> radiusExpression;
  if (predicate == "ST_Distance") {
    // ST_Distance returns a value, use it with a threshold.
    // For ST_Distance, we use a radius column instead of embedding the
    // threshold in the join condition.
    joinCondition =
        fmt::format("{}(probe_geom, build_geom) < radius", predicate);
    radiusColumn = "radius";
    radiusExpression = fmt::format(
        "CAST({} AS DOUBLE) AS radius",
        static_cast<double>(randInt(0, kMaxRadius)));
  } else {
    // Other predicates return boolean.
    joinCondition = fmt::format("{}(probe_geom, build_geom)", predicate);
  }

  // Create SpatialJoin plan with geometry conversion projections.
  auto spatialJoinPlan =
      test::PlanBuilder(planNodeIdGenerator)
          .values(probeInput)
          // Convert probe WKT strings to Geometry
          .project(
              {"probe_id",
               "ST_GeometryFromText(probe_geom_wkt) AS probe_geom",
               "probe_geom_wkt"})
          .spatialJoin(
              test::PlanBuilder(planNodeIdGenerator)
                  .values(buildInput)
                  // Convert build WKT strings to Geometry
                  .project(
                      radiusColumn.has_value()
                          ? std::vector<
                                std::
                                    string>{"build_id", "ST_GeometryFromText(build_geom_wkt) AS build_geom", "build_geom_wkt", radiusExpression.value()}
                          : std::vector<
                                std::
                                    string>{"build_id", "ST_GeometryFromText(build_geom_wkt) AS build_geom", "build_geom_wkt"})
                  .planNode(),
              joinCondition,
              "probe_geom",
              "build_geom",
              radiusColumn,
              {"probe_id", "probe_geom_wkt", "build_id", "build_geom_wkt"},
              joinType)
          .planNode();

  // Create equivalent NestedLoopJoin plan for comparison.
  auto nestedLoopJoinPlan =
      test::PlanBuilder(planNodeIdGenerator)
          .values(probeInput)
          // Convert probe WKT strings to Geometry
          .project(
              {"probe_id",
               "ST_GeometryFromText(probe_geom_wkt) AS probe_geom",
               "probe_geom_wkt"})
          .nestedLoopJoin(
              test::PlanBuilder(planNodeIdGenerator)
                  .values(buildInput)
                  // Convert build WKT strings to Geometry
                  .project(
                      radiusColumn.has_value()
                          ? std::vector<
                                std::
                                    string>{"build_id", "ST_GeometryFromText(build_geom_wkt) AS build_geom", "build_geom_wkt", radiusExpression.value()}
                          : std::vector<
                                std::
                                    string>{"build_id", "ST_GeometryFromText(build_geom_wkt) AS build_geom", "build_geom_wkt"})
                  .planNode(),
              {joinCondition},
              {"probe_id", "probe_geom_wkt", "build_id", "build_geom_wkt"},
              joinType)
          .planNode();

  LOG(INFO) << "Executing SpatialJoin plan...";
  const auto spatialJoinResult = execute(spatialJoinPlan);

  LOG(INFO) << "Executing NestedLoopJoin plan...";
  const auto nestedLoopJoinResult = execute(nestedLoopJoinPlan);

  // Compare SpatialJoin vs NestedLoopJoin results.
  auto result =
      test::assertEqualResults({nestedLoopJoinResult}, {spatialJoinResult});
  VELOX_CHECK(result, "SpatialJoin and NestedLoopJoin results don't match");

  LOG(INFO) << "SpatialJoin matches NestedLoopJoin.";
}

void SpatialJoinFuzzer::go() {
  VELOX_USER_CHECK(
      FLAGS_steps > 0 || FLAGS_duration_sec > 0,
      "Either --steps or --duration_sec needs to be greater than zero.");
  VELOX_USER_CHECK_GE(FLAGS_batch_size, 10, "Batch size must be at least 10.");

  const auto startTime = std::chrono::system_clock::now();

  while (!isDone(stats_.numIterations, startTime)) {
    LOG(WARNING) << "==============================> Started iteration "
                 << stats_.numIterations << " (seed: " << currentSeed_ << ")";

    // Pick join type.
    const auto joinType = pickJoinType();

    verify(joinType);

    LOG(WARNING) << "==============================> Done with iteration "
                 << stats_.numIterations;

    reSeed();
    ++stats_.numIterations;
  }

  LOG(INFO) << "Total iterations: " << stats_.numIterations;
}

} // namespace

void spatialJoinFuzzer(size_t seed) {
  SpatialJoinFuzzer(seed).go();
}

} // namespace facebook::velox::exec
