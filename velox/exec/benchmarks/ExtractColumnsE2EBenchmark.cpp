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

/// Benchmark comparing per-column extraction (sep) vs tiled column-major
/// extraction (tcm) with realistic schemas, shuffled rows, and batched output.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "velox/exec/RowContainer.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

struct SchemaSpec {
  const char* name;
  std::vector<TypePtr> keyTypes;
  std::vector<TypePtr> depTypes;
  int32_t numCols() const {
    return keyTypes.size() + depTypes.size();
  }
};

static std::vector<SchemaSpec> makeSchemas() {
  std::vector<SchemaSpec> specs;

  // Pure BIGINT micro-benchmark variants (2, 4, 8 columns).
  specs.push_back({"bigint2c", {BIGINT()}, {BIGINT()}});

  specs.push_back({"bigint4c", {BIGINT(), BIGINT()}, {BIGINT(), BIGINT()}});

  specs.push_back(
      {"bigint8c",
       {BIGINT(), BIGINT(), BIGINT(), BIGINT()},
       {BIGINT(), BIGINT(), BIGINT(), BIGINT()}});

  specs.push_back(
      {"mixWidth",
       {BOOLEAN(), INTEGER(), BIGINT()},
       {SMALLINT(), DOUBLE(), REAL(), BIGINT()}});

  specs.push_back(
      {"shortStr", {BIGINT(), VARCHAR()}, {INTEGER(), VARCHAR(), BIGINT()}});

  specs.push_back(
      {"longStr", {BIGINT(), VARCHAR()}, {VARCHAR(), DOUBLE(), INTEGER()}});

  specs.push_back(
      {"mixedAll",
       {BIGINT(), VARCHAR()},
       {INTEGER(), DOUBLE(), VARCHAR(), ARRAY(INTEGER())}});

  specs.push_back(
      {"wide16",
       {BIGINT(),
        BIGINT(),
        INTEGER(),
        BIGINT(),
        DOUBLE(),
        BIGINT(),
        INTEGER(),
        BIGINT()},
       {INTEGER(),
        DOUBLE(),
        SMALLINT(),
        BIGINT(),
        REAL(),
        INTEGER(),
        BIGINT(),
        DOUBLE()}});

  specs.push_back(
      {"tpchLine",
       {BIGINT(), INTEGER()},
       {BIGINT(),
        BIGINT(),
        DOUBLE(),
        DOUBLE(),
        DOUBLE(),
        DOUBLE(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR()}});

  specs.push_back(
      {"joinProbe",
       {BIGINT()},
       {BIGINT(), VARCHAR(), DOUBLE(), INTEGER(), VARCHAR(), BIGINT()}});

  specs.push_back(
      {"aggOutput",
       {BIGINT(), INTEGER(), SMALLINT(), BOOLEAN(), VARCHAR()},
       {BIGINT(), DOUBLE()}});

  specs.push_back({"globalAgg", {BIGINT(), BIGINT()}, {BIGINT(), DOUBLE()}});

  specs.push_back(
      {"allVarchar",
       {VARCHAR(), VARCHAR(), VARCHAR()},
       {VARCHAR(), VARCHAR(), VARCHAR()}});

  specs.push_back(
      {"wide30",
       {BIGINT(), BIGINT(), BIGINT()},
       {BIGINT(),  INTEGER(), DOUBLE(),  VARCHAR(), BIGINT(), INTEGER(),
        DOUBLE(),  BIGINT(),  VARCHAR(), DOUBLE(),  BIGINT(), INTEGER(),
        DOUBLE(),  VARCHAR(), BIGINT(),  INTEGER(), DOUBLE(), BIGINT(),
        VARCHAR(), DOUBLE(),  BIGINT(),  INTEGER(), DOUBLE(), VARCHAR(),
        BIGINT(),  INTEGER(), DOUBLE()}});

  return specs;
}

static RowTypePtr makeRowType(const SchemaSpec& spec) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (int32_t i = 0; i < static_cast<int32_t>(spec.keyTypes.size()); ++i) {
    names.push_back("k" + std::to_string(i));
    types.push_back(spec.keyTypes[i]);
  }
  for (int32_t i = 0; i < static_cast<int32_t>(spec.depTypes.size()); ++i) {
    names.push_back("d" + std::to_string(i));
    types.push_back(spec.depTypes[i]);
  }
  return ROW(std::move(names), std::move(types));
}

static RowVectorPtr generateBatch(
    const SchemaSpec& spec,
    int32_t numRows,
    memory::MemoryPool* pool) {
  auto rowType = makeRowType(spec);

  std::string sn(spec.name);
  bool longStrings =
      (sn == "longStr" || sn == "tpchLine" || sn == "allVarchar");

  VectorFuzzer::Options opts;
  opts.vectorSize = numRows;
  opts.nullRatio = 0.0;
  opts.stringLength = longStrings ? 50 : 8;
  opts.stringVariableLength = true;
  opts.containerLength = 3;
  opts.containerVariableLength = true;

  VectorFuzzer fuzzer(opts, pool, /*seed=*/42);
  return fuzzer.fuzzInputRow(rowType);
}

struct State {
  std::shared_ptr<memory::MemoryPool> pool;
  std::unique_ptr<RowContainer> container;
  std::vector<char*> rows;
  std::vector<column_index_t> cols;
  int32_t numCols{0};
  int32_t totalRows{0};

  std::vector<VectorPtr> resSep;
  std::vector<VectorPtr> resTcm;

  void init(
      const SchemaSpec& spec,
      int32_t numRows,
      int32_t outputBatch,
      const std::string& tag) {
    pool = memory::memoryManager()->addLeafPool(tag);
    numCols = spec.numCols();
    totalRows = numRows;

    container = std::make_unique<RowContainer>(
        spec.keyTypes, spec.depTypes, pool.get());

    constexpr int32_t kStoreBatch = 50000;
    rows.resize(numRows);
    for (int32_t off = 0; off < numRows; off += kStoreBatch) {
      int32_t n = std::min(kStoreBatch, numRows - off);
      auto batch = generateBatch(spec, n, pool.get());

      std::vector<char*> batchRows(n);
      for (int32_t r = 0; r < n; ++r) {
        batchRows[r] = container->newRow();
        rows[off + r] = batchRows[r];
      }
      for (int32_t c = 0; c < numCols; ++c) {
        DecodedVector dec(*batch->childAt(c), SelectivityVector(n));
        container->store(dec, folly::Range(batchRows.data(), n), c);
      }
    }

    // Shuffle to model real hash-table / sorted access patterns.
    std::mt19937 rng(12345);
    std::shuffle(rows.begin(), rows.end(), rng);

    cols.resize(numCols);
    std::iota(cols.begin(), cols.end(), 0);

    auto rowType = makeRowType(spec);

    resSep.resize(numCols);
    resTcm.resize(numCols);
    for (int32_t c = 0; c < numCols; ++c) {
      resSep[c] =
          BaseVector::create(rowType->childAt(c), outputBatch, pool.get());
      resTcm[c] =
          BaseVector::create(rowType->childAt(c), outputBatch, pool.get());
    }
  }

  void sep(int32_t batch) {
    for (int32_t off = 0; off < totalRows; off += batch) {
      int32_t n = std::min(batch, totalRows - off);
      for (int32_t c = 0; c < numCols; ++c) {
        container->extractColumn(rows.data() + off, n, cols[c], resSep[c]);
      }
    }
  }

  void tcm(int32_t batch) {
    for (int32_t off = 0; off < totalRows; off += batch) {
      int32_t n = std::min(batch, totalRows - off);
      container->extractColumns(rows.data() + off, n, cols, 0, resTcm);
    }
  }
};

namespace {

static const int32_t kContainerRows[] = {500000, 1000000, 2000000};
static const int32_t kOutputBatch = 4096;

std::string rowLabel(int32_t n) {
  if (n >= 1000000) {
    return std::to_string(n / 1000000) + "M";
  }
  return std::to_string(n / 1000) + "K";
}

std::vector<std::unique_ptr<State>> gStates;

void registerAll() {
  auto schemas = makeSchemas();
  int32_t count = 0;

  for (auto& spec : schemas) {
    for (int32_t nr : kContainerRows) {
      std::string sn(spec.name);

      // globalAgg only at 500K.
      if (sn == "globalAgg" && nr != 500000) {
        continue;
      }
      // Pure BIGINT micro-benchmarks run at 1M only.
      if ((sn == "bigint2c" || sn == "bigint4c" || sn == "bigint8c") &&
          nr != 1000000) {
        continue;
      }
      // Realistic schemas skip 1M (covered by 500K and 2M).
      if (nr == 1000000 && sn != "bigint2c" && sn != "bigint4c" &&
          sn != "bigint8c") {
        continue;
      }

      std::string tag = std::string(spec.name) + "_" +
          std::to_string(spec.numCols()) + "c_" + rowLabel(nr);

      auto st = std::make_unique<State>();
      fprintf(stderr, "  [%2d] %-40s ", ++count, tag.c_str());
      fflush(stderr);
      st->init(spec, nr, kOutputBatch, tag);
      fprintf(stderr, "ok\n");

      int32_t batch = kOutputBatch;
      auto* p = st.get();
      gStates.push_back(std::move(st));

      folly::addBenchmark(__FILE__, tag, [p, batch](unsigned n) {
        for (unsigned i = 0; i < n; ++i) {
          p->sep(batch);
        }
        return n;
      });
      folly::addBenchmark(__FILE__, "%" + tag + "_tcm", [p, batch](unsigned n) {
        for (unsigned i = 0; i < n; ++i) {
          p->tcm(batch);
        }
        return n;
      });
      folly::addBenchmark(__FILE__, "-", [](unsigned) { return 0; });
    }
  }
  fprintf(stderr, "  Total: %d scenarios\n\n", count);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::testingSetInstance({});
  registerAll();
  folly::runBenchmarks();
  gStates.clear();
  return 0;
}
