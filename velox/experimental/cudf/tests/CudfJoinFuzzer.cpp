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

#include "velox/experimental/cudf/tests/CudfJoinFuzzer.h"

#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/JoinFuzzerBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::cudf_velox::test {

namespace {

using namespace facebook::velox::exec;

/// CudfJoinFuzzer extends JoinFuzzerBase to test cuDF hash join
/// implementation. It restricts the join types and data types to those
/// supported by both cuDF and the DuckDB reference query runner.
class CudfJoinFuzzer : public JoinFuzzerBase {
 public:
  CudfJoinFuzzer(
      size_t initialSeed,
      std::unique_ptr<exec::test::ReferenceQueryRunner> referenceQueryRunner)
      : JoinFuzzerBase(
            initialSeed,
            std::move(referenceQueryRunner),
            "cudfJoinFuzzer") {}

 protected:
  std::vector<core::JoinType> getSupportedJoinTypes() const override {
    // Join types supported by both:
    // 1. CudfHashJoinProbe::isSupportedJoinType()
    // 2. DuckDB's SQL conversion (PrestoSqlPlanNodeVisitor)
    // Note: kRight and kRightSemiFilter are supported by cuDF but not by
    // DuckDB's SQL conversion.
    return {
        core::JoinType::kInner,
        core::JoinType::kLeft,
        core::JoinType::kFull,
        core::JoinType::kAnti,
        core::JoinType::kLeftSemiFilter,
    };
  }

  std::vector<TypePtr> getSupportedTypes() const override {
    // Types supported by cuDF for join keys and payload columns.
    // cuDF has more limited type support than CPU Velox.
    return {
        BOOLEAN(),
        TINYINT(),
        SMALLINT(),
        INTEGER(),
        BIGINT(),
        REAL(),
        DOUBLE(),
        VARCHAR(),
    };
  }

  void verify(core::JoinType joinType) override {
    const int numKeys = randInt(1, 5);
    std::vector<TypePtr> keyTypes = generateJoinKeyTypes(numKeys);

    std::vector<std::string> probeKeys = exec::test::makeNames("t", numKeys);
    std::vector<std::string> buildKeys = exec::test::makeNames("u", numKeys);

    auto probeInput = generateProbeInput(probeKeys, keyTypes);
    auto buildInput = generateBuildInput(probeInput, probeKeys, buildKeys);

    auto [convertedProbeInput, probeProjections] =
        referenceQueryRunner_->inputProjections(probeInput);
    auto [convertedBuildInput, buildProjections] =
        referenceQueryRunner_->inputProjections(buildInput);

    VELOX_CHECK(!convertedProbeInput.empty());
    VELOX_CHECK(!convertedBuildInput.empty());

    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Probe input: " << convertedProbeInput[0]->toString();
      for (const auto& v : convertedProbeInput) {
        VLOG(1) << std::endl << v->toString(0, v->size());
      }

      VLOG(1) << "Build input: " << convertedBuildInput[0]->toString();
      for (const auto& v : convertedBuildInput) {
        VLOG(1) << std::endl << v->toString(0, v->size());
      }
    }

    // Determine output columns based on join type.
    std::vector<std::string> outputColumns;
    if (core::isLeftSemiFilterJoin(joinType) ||
        core::isRightSemiFilterJoin(joinType) || core::isAntiJoin(joinType)) {
      if (core::isRightSemiFilterJoin(joinType)) {
        outputColumns = asRowType(buildInput[0]->type())->names();
      } else {
        outputColumns = asRowType(probeInput[0]->type())->names();
      }
    } else {
      auto combinedType = exec::test::concat(
          asRowType(probeInput[0]->type()), asRowType(buildInput[0]->type()));
      outputColumns = combinedType->names();
    }

    // Shuffle output columns.
    std::shuffle(outputColumns.begin(), outputColumns.end(), rng_);

    // Remove some output columns.
    const auto numOutput = randInt(1, outputColumns.size());
    outputColumns.resize(numOutput);

    // Build the hash join plan.
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan = exec::test::PlanBuilder(planNodeIdGenerator)
                    .values(convertedProbeInput)
                    .hashJoin(
                        probeKeys,
                        buildKeys,
                        exec::test::PlanBuilder(planNodeIdGenerator)
                            .values(convertedBuildInput)
                            .planNode(),
                        /*filter=*/"",
                        outputColumns,
                        joinType)
                    .planNode();

    LOG(INFO) << "Executing query plan: " << std::endl
              << plan->toString(true, true);

    exec::test::AssertQueryBuilder builder(plan);
    RowVectorPtr result = builder.maxDrivers(2).copyResults(pool_.get());

    LOG(INFO) << "Results: " << result->toString();
    if (VLOG_IS_ON(1)) {
      VLOG(1) << std::endl << result->toString(0, result->size());
    }

    exec::test::waitForAllTasksToBeDeleted();

    // Verify against reference.
    if (auto referenceResult = computeReferenceResults(
            plan, convertedProbeInput, convertedBuildInput)) {
      VELOX_CHECK(
          exec::test::assertEqualResults(
              referenceResult.value(), plan->outputType(), {result}),
          "Velox and Reference results don't match");

      LOG(INFO) << "Result matches with reference DB.";
      stats_.numVerified++;
    }
  }
};

} // namespace

void cudfJoinFuzzer(
    size_t seed,
    std::unique_ptr<exec::test::ReferenceQueryRunner> referenceQueryRunner) {
  CudfJoinFuzzer(seed, std::move(referenceQueryRunner)).go();
}

} // namespace facebook::velox::cudf_velox::test
