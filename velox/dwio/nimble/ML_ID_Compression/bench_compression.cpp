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

// bench_compression.cpp
//
// Phase-1 driver: encoded-size only, no timing.  Proves the encode path works
// end-to-end across the candidate encodings for ML ID columns.
//
// Usage:
//   bench_compression [--n=<rows>] [--seed=<seed>]
//                     [--dataset=<name>|all] [--encoder=<name>|all]
//                     [--output=<file.csv>]
//                     [--validate] [--dry_run]

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#endif
#include "folly/Random.h"
#include "velox/common/memory/Memory.h"

// ---------------------------------------------------------------------------
// CLI flags
// ---------------------------------------------------------------------------
DEFINE_int32(n, 100000, "Number of elements to generate per dataset");
DEFINE_int32(seed, 42, "RNG seed (currently informational; folly uses global)");
DEFINE_string(dataset, "all", "Dataset name to run, or 'all'");
DEFINE_string(encoder, "all", "Encoder name to run, or 'all'");
DEFINE_string(output, "", "Path for CSV output (stdout if empty)");
DEFINE_bool(validate, false, "Decode-and-compare round-trip after each encode");
DEFINE_bool(dry_run, false, "Print sweep plan and exit without encoding");

namespace facebook::nimble::mlidc {

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;
using namespace facebook::nimble::test;

// ---------------------------------------------------------------------------
// Type-erased encoder entry
// ---------------------------------------------------------------------------
struct EncoderEntry {
  std::string name;
  std::string family;
  std::string variant;
  bool isSequential;
  bool fastSkip;
  bool randomAccess;

  // Encode values into a fresh Buffer; return the encoded byte string.
  // The buffer is passed in so the caller can keep it alive.
  std::function<std::string(const Vector<int64_t>&, Encoding::Options&, Buffer&)>
      encodeFn;

  // Decode back from the encoded string; compare with original.
  // Returns true if round-trip matches.
  std::function<bool(
      const std::string& encoded,
      const Vector<int64_t>& original,
      Encoding::Options& options)>
      validateFn;
};

// ---------------------------------------------------------------------------
// Type-erased dataset entry
// ---------------------------------------------------------------------------
struct DatasetEntry {
  std::string name;
  std::function<Vector<int64_t>(int n)> generate;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Seed folly's random engine before generating data so results are
// reproducible across runs with the same --seed.
static void seedRng(int seed) {
  folly::Random::DefaultGenerator rng;
  rng.seed(static_cast<uint32_t>(seed));
  // folly::Random itself seeds from /dev/urandom; we just note the seed in
  // the output for reproducibility metadata.  Full seeding would require
  // threading a generator through the generators; omitted for Phase 1.
  (void)rng;
}

// ---------------------------------------------------------------------------
// Build the encoder table
// ---------------------------------------------------------------------------
static std::vector<EncoderEntry> buildEncoders() {
  std::vector<EncoderEntry> encoders;

  // Lambda helpers that capture the encoding type statically.
  auto addEncoder =
      [&](std::string name,
          std::string family,
          std::string variant,
          bool isSeq,
          bool fastSkip,
          bool randAccess,
          std::function<
              std::string(const Vector<int64_t>&, Encoding::Options&, Buffer&)>
              encFn,
          std::function<bool(
              const std::string&,
              const Vector<int64_t>&,
              Encoding::Options&)> valFn) {
        encoders.push_back(
            {std::move(name),
             std::move(family),
             std::move(variant),
             isSeq,
             fastSkip,
             randAccess,
             std::move(encFn),
             std::move(valFn)});
      };

  // Common validation lambda factory
  auto makeValidator = [](Encoding::Options opts) {
    return [opts](
               const std::string& encoded,
               const Vector<int64_t>& original,
               Encoding::Options& /*options*/) -> bool {
      auto& pool = benchmarkPool();
      EncodingFactory factory{opts};
      auto enc = factory.create(*pool, encoded, nullFactory());
      std::vector<int64_t> out(original.size());
      enc->materialize(static_cast<uint32_t>(original.size()), out.data());
      for (size_t i = 0; i < original.size(); ++i) {
        if (out[i] != original[i]) {
          return false;
        }
      }
      return true;
    };
  };

  // --- Trivial ---
  addEncoder(
      "Trivial",
      "Baseline",
      "trivial",
      /*isSeq=*/true,
      /*fastSkip=*/true,
      /*randAccess=*/false,
      [](const Vector<int64_t>& data, Encoding::Options& opts, Buffer& buf) {
        return std::string{
            Encoder<TrivialEncoding<int64_t>>::encode(
                buf, data, CompressionType::Uncompressed, opts)};
      },
      makeValidator({}));

  // --- FixedBitWidth ---
  addEncoder(
      "FixedBitWidth",
      "Baseline",
      "fbw",
      true,
      true,
      false,
      [](const Vector<int64_t>& data, Encoding::Options& opts, Buffer& buf) {
        return std::string{
            Encoder<FixedBitWidthEncoding<int64_t>>::encode(
                buf, data, CompressionType::Uncompressed, opts)};
      },
      makeValidator({}));

  // --- Dictionary ---
  addEncoder(
      "Dictionary",
      "Baseline",
      "dict",
      true,
      false,
      false,
      [](const Vector<int64_t>& data, Encoding::Options& opts, Buffer& buf) {
        return std::string{
            Encoder<DictionaryEncoding<int64_t>>::encode(
                buf, data, CompressionType::Uncompressed, opts)};
      },
      makeValidator({}));

  // --- RLE ---
  addEncoder(
      "RLE",
      "Baseline",
      "rle",
      true,
      true,
      false,
      [](const Vector<int64_t>& data, Encoding::Options& opts, Buffer& buf) {
        return std::string{
            Encoder<RLEEncoding<int64_t>>::encode(
                buf, data, CompressionType::Uncompressed, opts)};
      },
      makeValidator({}));

  // --- FPE variants (index types 0..3) ---
  const std::array<std::string, 4> fpeVariantNames = {
      "fpe_noindex", "fpe_pertier", "fpe_tagtag", "fpe_elias"};
  const std::array<bool, 4> fpeRandomAccess = {false, true, true, true};
  const std::array<bool, 4> fpeFastSkip = {false, true, true, true};

  for (int idx = 0; idx < 4; ++idx) {
    addEncoder(
        "FPE/" + fpeVariantNames[idx],
        "FrequencyPartition",
        fpeVariantNames[idx],
        /*isSeq=*/true,
        fpeFastSkip[idx],
        fpeRandomAccess[idx],
        [idx](
            const Vector<int64_t>& data,
            Encoding::Options& opts,
            Buffer& buf) {
          opts.frequencyPartitionIndex = static_cast<uint8_t>(idx);
          return std::string{
              Encoder<FrequencyPartitionEncoding<int64_t>>::encode(
                  buf, data, CompressionType::Uncompressed, opts)};
        },
        [idx](
            const std::string& encoded,
            const Vector<int64_t>& original,
            Encoding::Options& /*opts*/) -> bool {
          // NoIndex (idx 0) reorders by frequency with no restoration
          // index, so round-trip to original order is not possible.
          if (idx == 0) {
            return true;
          }
          auto& pool = benchmarkPool();
          Encoding::Options decodeOpts;
          decodeOpts.frequencyPartitionIndex = static_cast<uint8_t>(idx);
          FrequencyPartitionEncoding<int64_t> enc{
              *pool, encoded, nullFactory(), decodeOpts};
          std::vector<int64_t> out(original.size());
          enc.materialize(
              static_cast<uint32_t>(original.size()), out.data());
          for (size_t i = 0; i < original.size(); ++i) {
            if (out[i] != original[i]) {
              return false;
            }
          }
          return true;
        });
  }

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  // --- SubIntSplit with realNestedSelection ---
  // NOTE: EncodingFactory does NOT dispatch SubIntSplit, so validation must
  // go through test::Encoder<E>::createEncoding() rather than the factory.
  addEncoder(
      "SIS/realNested",
      "SubIntSplit",
      "real_nested",
      /*isSeq=*/false,
      /*fastSkip=*/false,
      /*randAccess=*/false,
      [](const Vector<int64_t>& data, Encoding::Options& opts, Buffer& buf) {
        return std::string{
            Encoder<SubIntSplitEncoding<int64_t>>::encode(
                buf,
                data,
                CompressionType::Uncompressed,
                opts,
                /*realNestedSelection=*/true)};
      },
      [](const std::string& encoded,
         const Vector<int64_t>& original,
         Encoding::Options& /*opts*/) -> bool {
        auto& pool = benchmarkPool();
        // Construct directly — EncodingFactory cannot dispatch SubIntSplit.
        SubIntSplitEncoding<int64_t> enc{*pool, encoded, nullFactory()};
        std::vector<int64_t> out(original.size());
        enc.materialize(static_cast<uint32_t>(original.size()), out.data());
        for (size_t i = 0; i < original.size(); ++i) {
          if (out[i] != original[i]) {
            return false;
          }
        }
        return true;
      });
#endif

  return encoders;
}

// ---------------------------------------------------------------------------
// Build the dataset table
// ---------------------------------------------------------------------------
static std::vector<DatasetEntry> buildDatasets() {
  return {
      {"UniformRandom",
       [](int n) {
         auto& pool = benchmarkPool();
         Vector<int64_t> data{pool.get()};
         data.resize(static_cast<uint32_t>(n));
         for (int i = 0; i < n; ++i) {
           data[i] = static_cast<int64_t>(folly::Random::secureRand64());
         }
         return data;
       }},
      {"LowCardinality16",
       [](int n) {
         return makeLowCardinality<int64_t>(16, static_cast<uint32_t>(n));
       }},
      {"RunLength",
       [](int n) {
         return makeRunLength<int64_t>(static_cast<uint32_t>(n));
       }},
      {"Increasing",
       [](int n) {
         return makeIncreasing<int64_t>(static_cast<uint32_t>(n));
       }},
      {"Narrow20",
       [](int n) {
         return makeNarrow<int64_t>(20, static_cast<uint32_t>(n));
       }},
  };
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------
static void writeCsvHeader(std::ostream& out) {
  out << "driver,dataset,encoding,family,variant,is_sequential,fast_skip,"
         "random_access,N,seed,payload_bytes,compression_ratio,"
         "bits_per_element,skipped\n";
}

static void writeCsvRow(
    std::ostream& out,
    const std::string& dataset,
    const EncoderEntry& enc,
    int n,
    int seed,
    size_t payloadBytes,
    double ratio,
    double bitsPerElem,
    bool skipped) {
  out << "bench_compression" << ',' << dataset << ',' << enc.name << ','
      << enc.family << ',' << enc.variant << ','
      << (enc.isSequential ? "true" : "false") << ','
      << (enc.fastSkip ? "true" : "false") << ','
      << (enc.randomAccess ? "true" : "false") << ',' << n << ',' << seed
      << ',' << payloadBytes << ',' << std::fixed << std::setprecision(4)
      << ratio << ',' << std::fixed << std::setprecision(4) << bitsPerElem
      << ',' << (skipped ? "true" : "false") << '\n';
}

} // namespace facebook::nimble::mlidc

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);
  facebook::velox::memory::MemoryManager::initialize({});

  using namespace facebook::nimble::mlidc;

  const int n = FLAGS_n;
  const int seed = FLAGS_seed;

  auto encoders = buildEncoders();
  auto datasets = buildDatasets();

  // Apply --encoder / --dataset filters
  if (FLAGS_encoder != "all") {
    encoders.erase(
        std::remove_if(
            encoders.begin(),
            encoders.end(),
            [&](const EncoderEntry& e) { return e.name != FLAGS_encoder; }),
        encoders.end());
    if (encoders.empty()) {
      std::cerr << "Unknown encoder: " << FLAGS_encoder << "\n";
      return 1;
    }
  }
  if (FLAGS_dataset != "all") {
    datasets.erase(
        std::remove_if(
            datasets.begin(),
            datasets.end(),
            [&](const DatasetEntry& d) { return d.name != FLAGS_dataset; }),
        datasets.end());
    if (datasets.empty()) {
      std::cerr << "Unknown dataset: " << FLAGS_dataset << "\n";
      return 1;
    }
  }

  // --dry-run: just print the sweep plan
  if (FLAGS_dry_run) {
    std::cout << "Sweep plan: " << datasets.size() << " dataset(s) x "
              << encoders.size() << " encoder(s), N=" << n << ", seed=" << seed
              << "\n\nDatasets:\n";
    for (const auto& d : datasets) {
      std::cout << "  " << d.name << "\n";
    }
    std::cout << "\nEncoders:\n";
    for (const auto& e : encoders) {
      std::cout << "  " << e.name << " [" << e.family << "/" << e.variant
                << "]\n";
    }
    return 0;
  }

  seedRng(seed);

  // Open output stream
  std::ofstream fileOut;
  std::ostream* csvOut = &std::cout;
  if (!FLAGS_output.empty()) {
    fileOut.open(FLAGS_output);
    if (!fileOut) {
      std::cerr << "Cannot open output file: " << FLAGS_output << "\n";
      return 1;
    }
    csvOut = &fileOut;
  }

  writeCsvHeader(*csvOut);

  int validateFailures = 0;

  for (const auto& ds : datasets) {
    // Generate data once per dataset
    auto data = ds.generate(n);
    const size_t rawBytes = static_cast<size_t>(n) * sizeof(int64_t);

    for (auto& enc : encoders) {
      // Each encode gets a fresh buffer (must outlive the string_view)
      facebook::nimble::Buffer buf{
          *facebook::nimble::benchmarks::benchmarkPool()};
      facebook::nimble::Encoding::Options opts;

      std::string encoded;
      bool skipped = false;

      try {
        encoded = enc.encodeFn(data, opts, buf);
      } catch (const std::exception& ex) {
        std::cerr << "[SKIP] " << enc.name << " / " << ds.name << ": "
                  << ex.what() << "\n";
        skipped = true;
      } catch (...) {
        std::cerr << "[SKIP] " << enc.name << " / " << ds.name
                  << ": unknown exception\n";
        skipped = true;
      }

      if (skipped) {
        writeCsvRow(
            *csvOut, ds.name, enc, n, seed, 0, 0.0, 0.0, /*skipped=*/true);
        continue;
      }

      const size_t payloadBytes = encoded.size();
      const double ratio =
          rawBytes > 0 ? static_cast<double>(payloadBytes) / rawBytes : 0.0;
      const double bitsPerElem =
          n > 0 ? (static_cast<double>(payloadBytes) * 8.0) / n : 0.0;

      // Optional round-trip validation
      if (FLAGS_validate) {
        bool ok = false;
        try {
          ok = enc.validateFn(encoded, data, opts);
        } catch (const std::exception& ex) {
          std::cerr << "[VALIDATE EXCEPTION] " << enc.name << " / " << ds.name
                    << ": " << ex.what() << "\n";
        }
        if (!ok) {
          std::cerr << "[VALIDATE FAIL] " << enc.name << " / " << ds.name
                    << "\n";
          ++validateFailures;
          // Exclude from output; mark skipped
          writeCsvRow(
              *csvOut,
              ds.name,
              enc,
              n,
              seed,
              0,
              0.0,
              0.0,
              /*skipped=*/true);
          continue;
        }
      }

      writeCsvRow(
          *csvOut,
          ds.name,
          enc,
          n,
          seed,
          payloadBytes,
          ratio,
          bitsPerElem,
          /*skipped=*/false);
    }
  }

  if (validateFailures > 0) {
    std::cerr << validateFailures
              << " validation failure(s); see rows with skipped=true.\n";
    return 2;
  }
  return 0;
}
