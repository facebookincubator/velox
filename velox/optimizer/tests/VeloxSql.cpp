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

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <sys/resource.h>
#include <sys/time.h>

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/Exchange.h"
#include "velox/optimizer/connectors/hive/LocalHiveConnectorMetadata.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Split.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/optimizer/Plan.h"
#include "velox/optimizer/VeloxHistory.h"
#include "velox/parse/QueryPlanner.h"
#include "velox/parse/TypeResolver.h"
#include "velox/runner/ConnectorSplitSource.h"
#include "velox/runner/LocalRunner.h"
#include "velox/runner/Schema.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorSaver.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::runner;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}

static bool validateDataFormat(const char* flagname, const std::string& value) {
  if ((value.compare("parquet") == 0) || (value.compare("dwrf") == 0)) {
    return true;
  }
  std::cout
      << fmt::format(
             "Invalid value for --{}: {}. Allowed values are [\"parquet\", \"dwrf\"]",
             flagname,
             value)
      << std::endl;
  return false;
}

int32_t printResults(const std::vector<RowVectorPtr>& results) {
  int32_t numRows = 0;
  std::cout << "Results:" << std::endl;
  bool printType = true;
  for (const auto& vector : results) {
    // Print RowType only once.
    if (printType) {
      std::cout << vector->type()->asRow().toString() << std::endl;
      printType = false;
    }
    for (vector_size_t i = 0; i < vector->size(); ++i) {
      std::cout << vector->toString(i) << std::endl;
      ++numRows;
    }
  }
  return numRows;
}
} // namespace

DEFINE_string(
    data_path,
    "",
    "Root path of data. Data layout must follow Hive-style partitioning. ");

DEFINE_string(ssd_path, "", "Directory for local SSD cache");
DEFINE_int32(ssd_cache_gb, 0, "Size of local SSD cache in GB");
DEFINE_int32(
    ssd_checkpoint_interval_gb,
    8,
    "Make a checkpoint after every n GB added to SSD cache");

DEFINE_int32(optimizer_trace, 0, "Optimizer trace level");

DEFINE_bool(print_plan, false, "Print optimizer results");

DEFINE_bool(print_stats, false, "print statistics");
DEFINE_bool(
    include_custom_stats,
    false,
    "Include custom statistics along with execution statistics");
DEFINE_int32(max_rows, 100, "Max number of printed result rows");
DEFINE_int32(num_drivers, 4, "Number of drivers");
DEFINE_int32(num_workers, 4, "Number of in-process workers");

DEFINE_string(data_format, "parquet", "Data format");
DEFINE_int32(num_splits_per_file, 10, "Number of splits per file");
DEFINE_int32(
    cache_gb,
    0,
    "GB of process memory for cache and query.. if "
    "non-0, uses mmap to allocator and in-process data cache.");
DEFINE_int32(num_repeats, 1, "Number of times to run --query");
DEFINE_string(
    query,
    "",
    "Text of query. If empty, reads ';' separated queries from standard input");

DEFINE_string(
    record,
    "",
    "Name of SQL file with a single query. Writes the "
    "output to <name>.ref for use with --check");
DEFINE_string(
    check,
    "",
    "Name of SQL file with a single query. Runs and "
    "compares with <name>.ref, previously recorded with --record");

DEFINE_validator(data_path, &notEmpty);
DEFINE_validator(data_format, &validateDataFormat);

struct RunStats {
  std::map<std::string, std::string> flags;
  int64_t micros{0};
  int64_t rawInputBytes{0};
  int64_t userNanos{0};
  int64_t systemNanos{0};
  std::string output;

  std::string toString(bool detail) {
    std::stringstream out;
    out << succinctNanos(micros * 1000) << " "
        << succinctBytes(rawInputBytes / (micros / 1000000.0)) << "/s raw, "
        << succinctNanos(userNanos) << " user " << succinctNanos(systemNanos)
        << " system (" << (100 * (userNanos + systemNanos) / (micros * 1000))
        << "%)";
    if (!flags.empty()) {
      out << ", flags: ";
      for (auto& pair : flags) {
        out << pair.first << "=" << pair.second << " ";
      }
    }
    out << std::endl << std::endl;
    if (detail) {
      out << std::endl << output << std::endl;
    }
    return out.str();
  }
};

class VeloxRunner {
 public:
  void initialize() {
    if (FLAGS_cache_gb) {
      memory::MemoryManagerOptions options;
      int64_t memoryBytes = FLAGS_cache_gb * (1LL << 30);
      options.useMmapAllocator = true;
      options.allocatorCapacity = memoryBytes;
      options.useMmapArena = true;
      options.mmapArenaCapacityRatio = 1;
      memory::MemoryManager::testingSetInstance(options);
      std::unique_ptr<cache::SsdCache> ssdCache;
      if (FLAGS_ssd_cache_gb) {
        constexpr int32_t kNumSsdShards = 16;
        cacheExecutor_ =
            std::make_unique<folly::IOThreadPoolExecutor>(kNumSsdShards);
        const cache::SsdCache::Config config(
            FLAGS_ssd_path,
            static_cast<uint64_t>(FLAGS_ssd_cache_gb) << 30,
            kNumSsdShards,
            cacheExecutor_.get(),
            static_cast<uint64_t>(FLAGS_ssd_checkpoint_interval_gb) << 30);
        ssdCache = std::make_unique<cache::SsdCache>(config);
      }

      cache_ = cache::AsyncDataCache::create(
          memory::memoryManager()->allocator(), std::move(ssdCache));
      cache::AsyncDataCache::setInstance(cache_.get());
    } else {
      memory::MemoryManager::testingSetInstance({});
    }

    rootPool_ = memory::memoryManager()->addRootPool("velox_sql");

    optimizerPool_ = rootPool_->addLeafChild("optimizer");
    schemaPool_ = rootPool_->addLeafChild("schema");
    checkPool_ = rootPool_->addLeafChild("check");

    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    parse::registerTypeResolver();
    filesystems::registerLocalFileSystem();
    parquet::registerParquetReaderFactory();
    dwrf::registerDwrfReaderFactory();
    exec::ExchangeSource::registerFactory(
        exec::test::createLocalExchangeSource);
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
    if (!isRegisteredNamedVectorSerde(VectorSerde::Kind::kPresto)) {
      serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
    }
    ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(8);
    std::unordered_map<std::string, std::string> connectorConfig;
    connectorConfig[connector::hive::HiveConfig::kLocalDataPath] =
        FLAGS_data_path;
    connectorConfig[connector::hive::HiveConfig::kLocalDefaultFileFormat] =
        FLAGS_data_format;
    auto config =
        std::make_shared<config::ConfigBase>(std::move(connectorConfig));
    connector::registerConnectorFactory(
        std::make_shared<connector::hive::HiveConnectorFactory>());
    connector_ =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, config, ioExecutor_.get());
    connector::registerConnector(connector_);

    std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
        connectorConfigs;
    auto copy = hiveConfig_;
    connectorConfigs[kHiveConnectorId] =
        std::make_shared<config::ConfigBase>(std::move(copy));

    schemaQueryCtx_ = core::QueryCtx::create(
        executor_.get(),
        core::QueryConfig(config_),
        std::move(connectorConfigs),
        cache::AsyncDataCache::getInstance(),
        rootPool_->shared_from_this(),
        spillExecutor_.get(),
        "schema");
    common::SpillConfig spillConfig;
    common::PrefixSortConfig prefixSortConfig;

    schemaRootPool_ = rootPool_->addAggregateChild("schemaRoot");
    connectorQueryCtx_ = std::make_shared<connector::ConnectorQueryCtx>(
        schemaPool_.get(),
        schemaRootPool_.get(),
        schemaQueryCtx_->connectorSessionProperties(kHiveConnectorId),
        &spillConfig,
        prefixSortConfig,
        std::make_unique<exec::SimpleExpressionEvaluator>(
            schemaQueryCtx_.get(), schemaPool_.get()),
        schemaQueryCtx_->cache(),
        "scan_for_schema",
        "schema",
        "N/a",
        0,
        schemaQueryCtx_->queryConfig().sessionTimezone());

    schema_ = std::make_shared<facebook::velox::runner::Schema>(connector_, "");

    planner_ = std::make_unique<core::DuckDbQueryPlanner>(optimizerPool_.get());
    auto& tables = dynamic_cast<connector::hive::LocalHiveConnectorMetadata*>(
                       connector_->metadata())
                       ->tables();
    for (auto& pair : tables) {
      planner_->registerTable(pair.first, pair.second->rowType());
    }
    planner_->registerTableScan(
        [this](
            const std::string& id,
            const std::string& name,
            const RowTypePtr& rowType,
            const std::vector<std::string>& columnNames) {
          return toTableScan(id, name, rowType, columnNames);
        });
    history_ = std::make_unique<facebook::velox::optimizer::VeloxHistory>();
    executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(
        FLAGS_num_drivers * 2 + 2);
    spillExecutor_ = std::make_shared<folly::IOThreadPoolExecutor>(4);
  }

  core::PlanNodePtr toTableScan(
      const std::string& id,
      const std::string& name,
      const RowTypePtr& rowType,
      const std::vector<std::string>& columnNames) {
    using namespace connector::hive;
    auto handle = std::make_shared<HiveTableHandle>(
        kHiveConnectorId, name, true, SubfieldFilters{}, nullptr);
    std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
        assignments;

    auto table = connector_->metadata()->findTable(name);
    for (auto i = 0; i < rowType->size(); ++i) {
      auto projectedName = rowType->nameOf(i);
      auto& columnName = columnNames[i];
      VELOX_CHECK(
          table->columnMap().find(columnName) != table->columnMap().end(),
          "No column {} in {}",
          columnName,
          name);
      assignments[projectedName] = std::make_shared<HiveColumnHandle>(
          columnName,
          HiveColumnHandle::ColumnType::kRegular,
          rowType->childAt(i),
          rowType->childAt(i));
    }
    return std::make_shared<core::TableScanNode>(
        id, rowType, handle, assignments);
  }

  void runInner(
      LocalRunner& runner,
      std::vector<RowVectorPtr>& result,
      RunStats& stats) {
    uint64_t micros = 0;
    {
      struct rusage start;
      getrusage(RUSAGE_SELF, &start);
      MicrosecondTimer timer(&micros);
      while (auto rows = runner.next()) {
        result.push_back(rows);
      }

      struct rusage final;
      getrusage(RUSAGE_SELF, &final);
      auto tvNanos = [](struct timeval tv) {
        return tv.tv_sec * 1000000000 + tv.tv_usec * 1000;
      };
      stats.userNanos = tvNanos(final.ru_utime) - tvNanos(start.ru_utime);
      stats.systemNanos = tvNanos(final.ru_stime) - tvNanos(start.ru_stime);
    }
    stats.micros = micros;
  }

  /// stores results and plans to 'ref', to be used with --check.
  void setRecordStream(std::ofstream* ref) {
    record_ = ref;
  }

  /// Compares results to data in 'ref'. 'ref' is produced with --record.
  void setCheckStream(std::ifstream* ref) {
    check_ = ref;
  }

  void run(const std::string& sql) {
    if (record_ || check_) {
      std::string error;
      std::string plan;
      std::vector<RowVectorPtr> result;
      run1(sql, nullptr, nullptr, &error);
      if (error.empty()) {
        run1(sql, &result, &plan, &error);
      }
      if (record_) {
        if (!error.empty()) {
          writeString(error, *record_);
        } else {
          writeString("", *record_);
          writeString(plan, *record_);
          writeVectors(result, *record_);
        }
      } else if (check_) {
        auto refError = readString(*check_);
        if (refError != error) {
          ++numFailed_;
          std::cerr << "Expected error "
                    << (refError.empty() ? std::string("no error") : refError)
                    << " got "
                    << (error.empty() ? std::string("no error") : error)
                    << std::endl;
          if (!refError.empty()) {
            readString(*check_);
            readVectors(*check_);
          }
          return;
        }
        if (!error.empty()) {
          // errors matched.
          return;
        }
        auto refPlan = readString(*check_);
        auto refResult = readVectors(*check_);
        bool planMiss = false;
        bool resultMiss = false;
        if (plan != refPlan) {
          std::cerr << "Plan mismatch: Expected " << refPlan << std::endl
                    << " got " << plan << std::endl;
          ++numPlanMismatch_;
          planMiss = true;
        }
        if (!assertEqualResults(refResult, result)) {
          ++numResultMismatch_;
          resultMiss = true;
        }
        if (!resultMiss && !planMiss) {
          ++numPassed_;
        } else {
          ++numFailed_;
        }
      }
    } else {
      run1(sql);
    }
  }

  std::string planCostString(
      const core::PlanNodeId& id,
      const facebook::velox::optimizer::Optimization::PlanCostMap& estimates) {
    auto it = estimates.find(id);
    if (it == estimates.end()) {
      return "";
    }
    std::stringstream out;
    for (auto& pair : it->second) {
      out << pair.first << ": " << pair.second.toString(true);
    }
    return out.str();
  }

  /// Runs a query and returns the result as a single vector in *resultVector,
  /// the plan text in *planString and the error message in *errorString.
  /// *errorString is not set if no error. Any of these may be nullptr.
  std::shared_ptr<LocalRunner> run1(
      const std::string& sql,
      std::vector<RowVectorPtr>* resultVector = nullptr,
      std::string* planString = nullptr,
      std::string* errorString = nullptr,
      std::vector<exec::TaskStats>* statsReturn = nullptr) {
    std::shared_ptr<LocalRunner> runner;
    std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
        connectorConfigs;
    auto copy = hiveConfig_;
    connectorConfigs[kHiveConnectorId] =
        std::make_shared<config::ConfigBase>(std::move(copy));
    ++queryCounter_;
    auto queryCtx = core::QueryCtx::create(
        executor_.get(),
        core::QueryConfig(config_),
        std::move(connectorConfigs),
        cache::AsyncDataCache::getInstance(),
        rootPool_->shared_from_this(),
        spillExecutor_.get(),
        fmt::format("query_{}", queryCounter_));

    // The default Locus for planning is the system and data of 'connector_'.
    optimizer::Locus locus(connector_->connectorId().c_str(), connector_.get());
    core::PlanNodePtr plan;
    try {
      plan = planner_->plan(sql);
    } catch (std::exception& e) {
      std::cerr << "parse error: " << e.what() << std::endl;
      if (errorString) {
        *errorString = fmt::format("Parse error: {}", e.what());
      }
      return nullptr;
    }
    facebook::velox::optimizer::Optimization::PlanCostMap estimates;
    MultiFragmentPlan::Options opts;
    opts.numWorkers = FLAGS_num_workers;
    opts.numDrivers = FLAGS_num_drivers;
    auto allocator =
        std::make_unique<HashStringAllocator>(optimizerPool_.get());
    auto context =
        std::make_unique<facebook::velox::optimizer::QueryGraphContext>(
            *allocator);
    facebook::velox::optimizer::queryCtx() = context.get();
    exec::SimpleExpressionEvaluator evaluator(
        queryCtx.get(), optimizerPool_.get());
    MultiFragmentPlanPtr fragmentedPlan;
    try {
      facebook::velox::optimizer::Schema veraxSchema(
          "test", schema_.get(), &locus);
      facebook::velox::optimizer::Optimization opt(
          *plan, veraxSchema, *history_, evaluator, FLAGS_optimizer_trace);
      auto best = opt.bestPlan();
      if (planString) {
        *planString = best->op->toString(true, false);
      }
      if (FLAGS_print_plan) {
        std::cout << "Plan: " << best->toString(true);
      }
      if (FLAGS_print_stats) {
        estimates = opt.planCostMap();
      }
      fragmentedPlan = opt.toVeloxPlan(best->op, opts);
    } catch (const std::exception& e) {
      facebook::velox::optimizer::queryCtx() = nullptr;
      std::cerr << "optimizer error: " << e.what() << std::endl;
      if (errorString) {
        *errorString = fmt::format("optimizer error: {}", e.what());
      }
      return nullptr;
    }
    facebook::velox::optimizer::queryCtx() = nullptr;
    RunStats runStats;
    try {
      runner = std::make_shared<LocalRunner>(
          fragmentedPlan,
          queryCtx,
          std::make_shared<runner::ConnectorSplitSourceFactory>());
      std::vector<RowVectorPtr> results;
      runInner(*runner, results, runStats);

      int numRows = printResults(results);
      if (resultVector) {
        *resultVector = results;
      }
      auto stats = runner->stats();
      if (statsReturn) {
        *statsReturn = stats;
      }
      auto& fragments = fragmentedPlan->fragments();
      for (int32_t i = fragments.size() - 1; i >= 0; --i) {
        for (auto& pipeline : stats[i].pipelineStats) {
          auto& first = pipeline.operatorStats[0];
          if (first.operatorType == "TableScan") {
            runStats.rawInputBytes += first.rawInputBytes;
          }
        }
        if (FLAGS_print_stats) {
          std::cout << "Fragment " << i << ":" << std::endl;
          std::cout << printPlanWithStats(
              *fragments[i].fragment.planNode,
              stats[i],
              FLAGS_include_custom_stats,
              [&](auto id) { return planCostString(id, estimates); });
          std::cout << std::endl;
        }
      }
      history_->recordVeloxExecution(nullptr, fragments, stats);
      std::cout << numRows << " rows " << runStats.toString(false) << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Query terminated with: " << e.what() << std::endl;
      if (errorString) {
        *errorString = fmt::format("Runtime error: {}", e.what());
      }
      waitForCompletion(runner);
      return nullptr;
    }
    waitForCompletion(runner);
    return runner;
  }

  void waitForCompletion(const std::shared_ptr<LocalRunner>& runner) {
    if (runner) {
      try {
        runner->waitForCompletion(50000);
      } catch (const std::exception& /*ignore*/) {
      }
    }
  }

  /// Returns exit status for run. 0 is passed, 1 is plan differences only, 2 is
  /// result differences.
  int32_t checkStatus() {
    std::cerr << numPassed_ << " passed " << numFailed_ << " failed "
              << numPlanMismatch_ << " plan mismatch " << numResultMismatch_
              << " result mismatch" << std::endl;
    if (!numFailed_) {
      return 0;
    }
    return numResultMismatch_ ? 2 : 1;
  }

 private:
  template <typename T>
  static void write(const T& value, std::ostream& out) {
    out.write((char*)&value, sizeof(T));
  }

  template <typename T>
  static T read(std::istream& in) {
    T value;
    in.read((char*)&value, sizeof(T));
    return value;
  }

  static std::string readString(std::istream& in) {
    auto len = read<int32_t>(in);
    std::string result;
    result.resize(len);
    in.read(result.data(), result.size());
    return result;
  }

  static void writeString(const std::string& string, std::ostream& out) {
    write<int32_t>(string.size(), out);
    out.write(string.data(), string.size());
  }

  std::vector<RowVectorPtr> readVectors(std::istream& in) {
    auto size = read<int32_t>(in);
    std::vector<RowVectorPtr> result(size);
    for (auto i = 0; i < size; ++i) {
      result[i] = std::dynamic_pointer_cast<RowVector>(
          restoreVector(in, checkPool_.get()));
    }
    return result;
  }

  void writeVectors(std::vector<RowVectorPtr>& vectors, std::ostream& out) {
    write<int32_t>(vectors.size(), out);
    for (auto& vector : vectors) {
      saveVector(*vector, out);
    }
  }

  int32_t printResults(const std::vector<RowVectorPtr>& results) {
    std::cout << "Results:" << std::endl;
    bool printType = true;
    int32_t numRows = 0;
    for (auto vectorIndex = 0; vectorIndex < results.size(); ++vectorIndex) {
      const auto& vector = results[vectorIndex];
      // Print RowType only once.
      if (printType) {
        std::cout << vector->type()->asRow().toString() << std::endl;
        printType = false;
      }
      for (vector_size_t i = 0; i < vector->size(); ++i) {
        std::cout << vector->toString(i) << std::endl;
        if (++numRows >= FLAGS_max_rows) {
          int32_t numLeft = (vector->size() - (i - 1));
          ++vectorIndex;
          for (; vectorIndex < results.size(); ++vectorIndex) {
            numLeft += results[vectorIndex]->size();
          }
          if (numLeft) {
            std::cout << fmt::format("[Omitted {} more rows.", numLeft)
                      << std::endl;
          }
          return numRows + numLeft;
        }
      }
    }
    return numRows;
  }

  std::shared_ptr<memory::MemoryAllocator> allocator_;
  std::shared_ptr<cache::AsyncDataCache> cache_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> optimizerPool_;
  std::shared_ptr<memory::MemoryPool> schemaPool_;
  std::shared_ptr<memory::MemoryPool> schemaRootPool_;
  std::shared_ptr<memory::MemoryPool> checkPool_;
  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
  std::unique_ptr<folly::IOThreadPoolExecutor> cacheExecutor_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::shared_ptr<folly::IOThreadPoolExecutor> spillExecutor_;
  std::shared_ptr<core::QueryCtx> schemaQueryCtx_;
  std::shared_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  std::shared_ptr<connector::Connector> connector_;
  std::shared_ptr<runner::Schema> schema_;
  std::unique_ptr<facebook::velox::optimizer::VeloxHistory> history_;
  std::unique_ptr<core::DuckDbQueryPlanner> planner_;
  std::unordered_map<std::string, std::string> config_;
  std::unordered_map<std::string, std::string> hiveConfig_;
  std::ofstream* record_{nullptr};
  std::ifstream* check_{nullptr};
  int32_t numPassed_{0};
  int32_t numFailed_{0};
  int32_t numPlanMismatch_{0};
  int32_t numResultMismatch_{0};
  int32_t queryCounter_{0};
};

std::string readCommand(std::istream& in, bool& end) {
  std::string line;
  std::stringstream command;
  end = false;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == ';') {
      command << line.substr(0, line.size() - 1);
      return command.str();
    }
    command << line << std::endl;
  }
  end = true;
  return "";
}

void readCommands(
    VeloxRunner& runner,
    const std::string& prompt,
    std::istream& in) {
  for (;;) {
    std::cout << prompt;
    bool end;
    std::string command = readCommand(in, end);
    if (end) {
      break;
    }
    if (command.empty()) {
      continue;
    }
    auto cstr = command.c_str();
    char* flag = nullptr;
    char* value = nullptr;
    if (sscanf(cstr, "flag %ms = %ms", &flag, &value) == 2) {
      std::cout << gflags::SetCommandLineOption(flag, value);
      free(flag);
      free(value);
      continue;
    }
    runner.run(command);
  }
}

void recordQueries(VeloxRunner& runner) {
  std::ifstream in(FLAGS_record);
  std::ofstream ref;
  ref.open(FLAGS_record + ".ref", std::ios_base::out | std::ios_base::trunc);
  runner.setRecordStream(&ref);
  readCommands(runner, "", in);
}

void checkQueries(VeloxRunner& runner) {
  std::ifstream in(FLAGS_check);
  std::ifstream ref(FLAGS_check + ".ref");
  runner.setCheckStream(&ref);
  readCommands(runner, "", in);
  exit(runner.checkStatus());
}

std::string sevenBit(std::string& in) {
  for (auto i = 0; i < in.size(); ++i) {
    if ((uint8_t)in[i] > 127) {
      in[i] = ' ';
    }
  }
  return in;
}

int main(int argc, char** argv) {
  std::string kUsage(
      "Velox local SQL command line. Run 'velox_sql --help' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init(&argc, &argv, false);
  VeloxRunner runner;
  try {
    runner.initialize();
    if (!FLAGS_query.empty()) {
      runner.run(FLAGS_query);
    } else if (!FLAGS_record.empty()) {
      recordQueries(runner);
    } else if (!FLAGS_check.empty()) {
      checkQueries(runner);
    } else {
      std::cout
          << "Velox SQL. Type statement and end with ;. flag name = value; sets a gflag."
          << std::endl;
      readCommands(runner, "SQL> ", std::cin);
    }
  } catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(-1);
  }
  return 0;
}
