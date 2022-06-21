#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <velox/core/PlanFragment.h>
#include <velox/core/PlanNode.h>
#include <optional>
#include "common/init/Init.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempFilePath.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"

using namespace facebook::velox;

namespace {

DEFINE_bool(print_output, false, "Print Output (true, false)");

DEFINE_int32(num_rows, 1000, "Number of rows in the input data set.");

class RandomSamplingBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  explicit RandomSamplingBenchmark()
      : FunctionBenchmarkBase(),
        hiveConnectorId_("test-hive"),
        file_(exec::test::TempFilePath::create()) {
    rowType_ =
        ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
            {BIGINT(),
             INTEGER(),
             SMALLINT(),
             REAL(),
             DOUBLE(),
             VARCHAR(),
             TINYINT()});
    std::vector<RowVectorPtr> data = makeVectors(10, FLAGS_num_rows);
    writeToFile(data);
    setup();
  }

  size_t runEval(
      std::optional<std::string> query,
      const bool createFilterNode,
      int times) {
    const core::PlanFragment planFragment = tableScan(query, createFilterNode);

    auto runTask = [&]() {
      auto tableScanTask = createTask(planFragment);
      runTaskSync(tableScanTask);
    };

    for (auto i = 0; i < times; i++) {
      runTask();
    }

    return times;
  }

 private:
  void setup() {
    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(hiveConnectorId_, nullptr);
    connector::registerConnector(hiveConnector);

    filesystems::registerLocalFileSystem();
    dwrf::registerDwrfReaderFactory();

    serializer::presto::PrestoVectorSerde::registerVectorSerde();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
  }

  std::vector<RowVectorPtr> makeVectors(
      const int32_t count,
      const int32_t rowsPerVector) const {
    std::vector<RowVectorPtr> vectors;
    vectors.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          test::BatchMaker::createBatch(rowType_, rowsPerVector, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  void writeToFile(
      const std::vector<RowVectorPtr>& vectors,
      std::shared_ptr<dwrf::Config> config =
          std::make_shared<dwrf::Config>()) const {
    const std::string kWriter = "tablescan.writer";
    dwrf::WriterOptions options;
    options.config = config;
    options.schema = vectors[0]->type();
    auto sink = std::make_unique<dwio::common::FileSink>(file_->path);
    dwrf::Writer writer{
        options,
        std::move(sink),
        pool_->addChild(kWriter, std::numeric_limits<int64_t>::max())};

    for (size_t i = 0; i < vectors.size(); ++i) {
      writer.write(vectors[i]);
    }
    writer.close();
  }

  core::PlanFragment tableScan(
      const std::optional<std::string>& query,
      const bool createFilterNode) const {
    auto builder = exec::test::PlanBuilder();
    if (query.has_value()) {
      if (createFilterNode) {
        builder = builder.tableScan(rowType_).filter(query.value());
      } else {
        builder = builder.tableScan(rowType_, {}, query.value());
      }
    } else {
      builder = builder.tableScan(rowType_);
    }
    auto planFragment = builder.planFragment();
    return planFragment;
  }

  std::shared_ptr<exec::Task> createTask(core::PlanFragment planFragment) {
    auto task = std::make_shared<exec::Task>(
        "tablescan.task",
        planFragment,
        0,
        core::QueryCtx::createForTest(),
        [](RowVectorPtr vector, exec::ContinueFuture*) {
          if (vector && FLAGS_print_output) {
            LOG(INFO) << "Vector available with size " << vector->size();
            for (size_t i = 0; i < vector->size(); ++i) {
              LOG(INFO) << vector->toString(i);
            }
          }
          return exec::BlockingReason::kNotBlocked;
        });

    auto connectorSplit = std::make_shared<connector::hive::HiveConnectorSplit>(
        hiveConnectorId_, "file:" + file_->path, dwio::common::FileFormat::ORC);
    task->addSplit("0", exec::Split{connectorSplit});
    task->noMoreSplits("0");

    return task;
  }

  void runTaskSync(std::shared_ptr<exec::Task> task) {
    exec::Task::start(task, 1);
    auto& inlineExecutor = folly::QueuedImmediateExecutor::instance();
    task->stateChangeFuture(0).via(&inlineExecutor).wait();
  }

  std::shared_ptr<const RowType> rowType_;
  std::string hiveConnectorId_;
  std::shared_ptr<exec::test::TempFilePath> file_;
};

std::unique_ptr<RandomSamplingBenchmark> benchmark;

BENCHMARK_MULTI(runTableScan, n) {
  return benchmark->runEval(std::nullopt, false, n);
}

BENCHMARK_MULTI(runTableScanWithFilter, n) {
  return benchmark->runEval(std::make_optional("RANDOM() < 0.1"), false, n);
}

BENCHMARK_MULTI(runTableScanFollowedByFilter, n) {
  return benchmark->runEval(std::make_optional("RANDOM() < 0.1"), true, n);
}

} // namespace

int main(int argc, char** argv) {
  folly::init(&argc, &argv);

  benchmark = std::make_unique<RandomSamplingBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
