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
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzerRunner.h"

#include <chrono>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzer.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"

// Defined here and only here: the runner library, the gtest and the standalone
// binary all link this translation unit, and a second definition would abort
// gflags at startup on duplicate registration. Every flag carries the
// nimble_writer_fuzzer_ prefix for the same reason -- bare names like "seed"
// and "batch_size" are already defined by other fbcode binaries, and any future
// binary linking both would abort.
DEFINE_uint64(
    nimble_writer_fuzzer_seed,
    0,
    "Fuzzer seed. When unset a random seed is chosen and logged.");
DEFINE_int32(
    nimble_writer_fuzzer_duration_sec,
    30,
    "How long to run the fuzzer loop. At least one iteration always runs.");
DEFINE_int32(
    nimble_writer_fuzzer_max_schema_depth,
    3,
    "Maximum nesting depth of fuzzed schemas.");
DEFINE_int32(nimble_writer_fuzzer_batch_size, 200, "Rows per written batch.");
DEFINE_int32(nimble_writer_fuzzer_num_batches, 3, "Batches written per file.");
DEFINE_bool(
    nimble_writer_fuzzer_require_coverage,
    false,
    "Fail the run if any candidate encoding or required chunk-stats metadata "
    "shape was not exercised. Off by default so short local runs do not fail "
    "on a shape they did not happen to draw; enabled in CI, where the run is "
    "long enough to cover every required case.");

namespace facebook::nimble::fuzzer {

void setUpFuzzerEnvironments() {
  registerSelectiveNimbleReaderFactory();
}

void runNimbleWriterFuzzer() {
  if (gflags::GetCommandLineFlagInfoOrDie("nimble_writer_fuzzer_seed")
          .is_default) {
    FLAGS_nimble_writer_fuzzer_seed = std::random_device{}();
    // WARNING, not INFO: the Cogwheel job runs with --minloglevel=1, and a
    // randomly chosen seed that is not in the log cannot be reproduced. Same
    // reasoning for the coverage report below.
    LOG(WARNING) << "Use generated random seed "
                 << FLAGS_nimble_writer_fuzzer_seed;
  }

  NIMBLE_USER_CHECK_GT(
      FLAGS_nimble_writer_fuzzer_max_schema_depth,
      0,
      "--nimble_writer_fuzzer_max_schema_depth must be positive.");
  NIMBLE_USER_CHECK_GT(
      FLAGS_nimble_writer_fuzzer_batch_size,
      0,
      "--nimble_writer_fuzzer_batch_size must be positive.");
  NIMBLE_USER_CHECK_GT(
      FLAGS_nimble_writer_fuzzer_num_batches,
      0,
      "--nimble_writer_fuzzer_num_batches must be positive.");

  auto rootPool = velox::memory::memoryManager()->addRootPool(
      "NimbleWriterFuzzer", velox::memory::kMaxMemory);

  NimbleWriterFuzzerOptions options;
  options.seed = FLAGS_nimble_writer_fuzzer_seed;
  options.maxSchemaDepth = FLAGS_nimble_writer_fuzzer_max_schema_depth;
  options.batchSize = FLAGS_nimble_writer_fuzzer_batch_size;
  options.numBatches = FLAGS_nimble_writer_fuzzer_num_batches;

  NimbleWriterFuzzer fuzzer(options, *rootPool);
  const auto deadline = std::chrono::system_clock::now() +
      std::chrono::seconds(FLAGS_nimble_writer_fuzzer_duration_sec);
  int iteration = 0;
  auto reportCoverage = [&]() {
    LOG(WARNING) << "Completed " << iteration << " iterations.";
    fuzzer.logCoverage();
    fuzzer.logPairCoverage();
    fuzzer.logChunkStatsCoverage();
  };

  // Catch and rethrow rather than a scope guard: an exception that escapes
  // main is never caught, so std::terminate runs without unwinding the stack
  // and no destructor-based reporting would fire. On a failure the tally is
  // the main clue for how much had actually been exercised, and that is
  // exactly the run where losing it hurts most.
  try {
    // Do-while, not while: a zero or negative duration would otherwise run no
    // iterations at all and report a passing test that verified nothing.
    do {
      LOG(INFO) << "Starting iteration " << iteration
                << ", seed=" << fuzzer.seed();
      fuzzer.run();
      fuzzer.reSeed();
      ++iteration;
    } while (std::chrono::system_clock::now() < deadline);
  } catch (...) {
    reportCoverage();
    throw;
  }
  reportCoverage();

  if (FLAGS_nimble_writer_fuzzer_require_coverage) {
    const auto unapplied = fuzzer.unappliedEncodings();
    if (!unapplied.empty()) {
      std::vector<std::string> details;
      details.reserve(unapplied.size());
      for (const auto encodingType : unapplied) {
        const auto entry = fuzzer.coverage().find(encodingType);
        std::string reason;
        if (entry == fuzzer.coverage().end()) {
          reason = "never attempted";
        } else {
          reason = fmt::format(
              "{} files offered ({} forced repair files), never applied",
              entry->second.numFilesOffered,
              entry->second.numForcedFilesWritten);
        }
        details.push_back(
            fmt::format("{} ({})", toString(encodingType), std::move(reason)));
      }
      NIMBLE_FAIL(
          "{} of {} candidate encodings were never applied after {} iterations: {}. "
          "Either the data shaping no longer produces inputs they accept, or they "
          "became unselectable.",
          unapplied.size(),
          allCandidateEncodings().size(),
          iteration,
          fmt::join(details, ", "));
    }

    // Checked after the per-encoding gate: an encoding that was never applied
    // at all also shows up here once per data type it accepts, and the
    // per-encoding message names the cause far more directly.
    const auto unappliedPairs = fuzzer.unappliedPairs();
    if (!unappliedPairs.empty()) {
      std::vector<std::string> details;
      details.reserve(unappliedPairs.size());
      for (const auto& [dataType, encodingType] : unappliedPairs) {
        details.push_back(
            fmt::format("{}/{}", toString(dataType), toString(encodingType)));
      }
      NIMBLE_FAIL(
          "{} (data type, encoding) pairs were never applied after {} iterations: {}. "
          "Each pair names a stream type the encoding accepts but was never "
          "actually used on, which the per-encoding tally cannot show.",
          unappliedPairs.size(),
          iteration,
          fmt::join(details, ", "));
    }

    const auto uncoveredChunkStatsShapes = fuzzer.uncoveredChunkStatsShapes();
    NIMBLE_CHECK(
        uncoveredChunkStatsShapes.empty(),
        "Chunk-stats metadata coverage is incomplete after {} iterations: {}.",
        iteration,
        fmt::join(uncoveredChunkStatsShapes, ", "));
  }
}

} // namespace facebook::nimble::fuzzer
