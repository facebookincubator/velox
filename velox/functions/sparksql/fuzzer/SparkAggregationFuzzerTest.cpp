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

#include <boost/random/uniform_int_distribution.hpp>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <array>
#include <optional>
#include <unordered_set>

#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/fuzzer/AggregationFuzzerOptions.h"
#include "velox/exec/fuzzer/AggregationFuzzerRunner.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/TransformResultVerifier.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/functions/sparksql/fuzzer/SparkQueryRunner.h"
#include "velox/serializers/CompactRowSerializer.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/UnsafeRowSerializer.h"

DECLARE_int32(batch_size);
DECLARE_int32(num_batches);

DEFINE_int64(
    seed,
    0,
    "Initial seed for random number generator used to reproduce previous "
    "results (0 means start with random seed).");

DEFINE_string(
    only,
    "",
    "If specified, Fuzzer will only choose functions from "
    "this comma separated list of function names "
    "(e.g: --only \"min\" or --only \"sum,avg\").");

DEFINE_int64(allocator_capacity, 8L << 30, "Allocator capacity in bytes.");

DEFINE_int64(arbitrator_capacity, 6L << 30, "Arbitrator capacity in bytes.");

namespace {

// Generates valid query-wide constant parameters for count_min_sketch.
class CountMinSketchInputGenerator final
    : public facebook::velox::exec::test::InputGenerator {
 public:
  std::vector<facebook::velox::VectorPtr> generate(
      const std::vector<facebook::velox::TypePtr>& types,
      facebook::velox::VectorFuzzer& fuzzer,
      facebook::velox::FuzzerGenerator& rng,
      facebook::velox::memory::MemoryPool* pool) override {
    VELOX_CHECK_EQ(types.size(), 4);

    if (!epsilon_.has_value()) {
      static constexpr std::array<double, 3> kEpsilons{0.1, 0.2, 0.5};
      static constexpr std::array<double, 3> kConfidences{0.5, 0.75, 0.95};
      epsilon_ = kEpsilons[boost::random::uniform_int_distribution<uint32_t>(
          0, static_cast<uint32_t>(kEpsilons.size() - 1))(rng)];
      confidence_ =
          kConfidences[boost::random::uniform_int_distribution<uint32_t>(
              0, static_cast<uint32_t>(kConfidences.size() - 1))(rng)];
      seed_ =
          boost::random::uniform_int_distribution<int32_t>(-1'000, 1'000)(rng);
    }

    const auto size = fuzzer.getOptions().vectorSize;
    const auto seed = types[3]->isInteger()
        ? facebook::velox::variant(seed_.value())
        : facebook::velox::variant(static_cast<int64_t>(seed_.value()));
    return {
        fuzzer.fuzz(types[0], size),
        facebook::velox::BaseVector::createConstant(
            types[1], epsilon_.value(), size, pool),
        facebook::velox::BaseVector::createConstant(
            types[2], confidence_.value(), size, pool),
        facebook::velox::BaseVector::createConstant(types[3], seed, size, pool),
    };
  }

  void reset() override {
    epsilon_.reset();
    confidence_.reset();
    seed_.reset();
  }

 private:
  std::optional<double> epsilon_;
  std::optional<double> confidence_;
  std::optional<int32_t> seed_;
};

} // namespace

int main(int argc, char** argv) {
  facebook::velox::functions::aggregate::sparksql::registerAggregateFunctions(
      "", false);
  facebook::velox::parquet::registerParquetWriterFactory();

  ::testing::InitGoogleTest(&argc, argv);

  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for better debugging
  // experience, and initialize glog and gflags.
  folly::Init init(&argc, &argv);

  facebook::velox::functions::prestosql::registerInternalFunctions();
  if (!facebook::velox::isRegisteredNamedVectorSerde("Presto")) {
    facebook::velox::serializer::presto::PrestoVectorSerde::
        registerNamedVectorSerde();
  }
  if (!facebook::velox::isRegisteredNamedVectorSerde("CompactRow")) {
    facebook::velox::serializer::CompactRowVectorSerde::
        registerNamedVectorSerde();
  }
  if (!facebook::velox::isRegisteredNamedVectorSerde("UnsafeRow")) {
    facebook::velox::serializer::spark::UnsafeRowVectorSerde::
        registerNamedVectorSerde();
  }
  // Must install a real arbitrator. With the default options the manager gets
  // a NoopArbitrator, so the test-only spill hooks reach
  // memory::testingRunArbitration() and reclaim nothing, which leaves hash
  // aggregation -- the one operator here that spills only under arbitration --
  // never spilling.
  facebook::velox::exec::test::setupMemory(
      FLAGS_allocator_capacity, FLAGS_arbitrator_capacity);

  // Spark reference execution uses gRPC and can be sensitive to large
  // payloads. Keep generated input sizes modest to reduce transport and
  // memory pressure.
  FLAGS_batch_size = 40;
  FLAGS_num_batches = 4;

  // Spark does not provide user-accessible aggregate functions with the
  // following names.
  std::unordered_set<std::string> skipFunctions = {
      "bloom_filter_agg",
      // Velox registers a 2-arg collect_set(T, boolean) signature that Spark
      // doesn't support. The fuzzer may pick this signature and fail.
      "collect_set",
      "first_ignore_null",
      "last_ignore_null",
      "regr_replacement",
      // https://github.com/facebookincubator/velox/issues/17124
      // Correctness mismatches and OOM during KLL sketch operations.
      "approx_percentile",
  };

  using facebook::velox::exec::test::TransformResultVerifier;

  auto makeArrayVerifier = []() {
    return TransformResultVerifier::create("\"$internal$canonicalize\"({})");
  };

  // The results of the following functions depend on the order of input
  // rows. For some functions, the result can be transformed to a value that
  // doesn't depend on the order of inputs. If such transformation exists, it
  // can be specified to be used for results verification. If no transformation
  // is specified, results are not verified.
  std::unordered_map<
      std::string,
      std::shared_ptr<facebook::velox::exec::test::ResultVerifier>>
      customVerificationFunctions = {
          {"last", nullptr},
          {"last_ignore_null", nullptr},
          {"first", nullptr},
          {"first_ignore_null", nullptr},
          {"max_by", nullptr},
          {"min_by", nullptr},
          // If multiple values have the same greatest frequency, the return
          // value is indeterminate.
          {"mode", nullptr},
          {"skewness", nullptr},
          {"kurtosis", nullptr},
          {"collect_list", makeArrayVerifier()},
          {"collect_set", makeArrayVerifier()},
          // Nested nulls are handled as values in Spark. But nested nulls
          // comparison always generates null in DuckDB.
          {"min", nullptr},
          {"max", nullptr},
      };

  size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool{
      facebook::velox::memory::memoryManager()->addRootPool()};
  auto sparkQueryRunner = std::make_unique<
      facebook::velox::functions::sparksql::fuzzer::SparkQueryRunner>(
      rootPool.get(), "localhost:15002", "fuzzer", "aggregate");

  using Runner = facebook::velox::exec::test::AggregationFuzzerRunner;
  using Options = facebook::velox::exec::test::AggregationFuzzerOptions;

  Options options;
  options.onlyFunctions = FLAGS_only;
  options.skipFunctions = skipFunctions;
  options.customVerificationFunctions = customVerificationFunctions;
  options.customInputGenerators = {
      {"count_min_sketch", std::make_shared<CountMinSketchInputGenerator>()},
  };
  options.orderableGroupKeys = true;
  options.timestampPrecision =
      facebook::velox::VectorFuzzer::Options::TimestampPrecision::kMicroSeconds;
  options.hiveConfigs = {
      {facebook::velox::connector::hive::HiveConfig::kReadTimestampUnit, "6"}};
  return Runner::run(initialSeed, std::move(sparkQueryRunner), options);
}
