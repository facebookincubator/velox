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

#include <folly/init/Init.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseManager.h>
#include <folly/json.h>
#include <folly/portability/GFlags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <proxygen/httpserver/HTTPServer.h>
#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/RequestHandlerFactory.h>
#include <proxygen/httpserver/ResponseBuilder.h>
#include <memory>
#include <string>
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/runner/LocalRunner.h"

using namespace facebook::velox;
using namespace proxygen;

DEFINE_int32(port, 9090, "Port to listen on");
DEFINE_string(
    registry,
    "presto",
    "Function registry to use for query evaluation. Currently supported values are "
    "presto and spark. Default is presto.");
DEFINE_int32(num_workers, 4, "Number of workers to use for query execution");
DEFINE_int32(num_drivers, 2, "Number of drivers per worker");

static bool validateRegistry(const char* flagName, const std::string& value) {
  static const std::unordered_set<std::string> kRegistries = {
      "presto", "spark"};
  if (kRegistries.count(value) != 1) {
    std::cerr << "Invalid value for --" << flagName << ": " << value << ". ";
    std::cerr << "Valid values are: " << folly::join(", ", kRegistries) << "."
              << std::endl;
    return false;
  }
  if (value == "spark") {
    functions::sparksql::registerFunctions("");
  } else if (value == "presto") {
    functions::prestosql::registerAllScalarFunctions();
  }

  return true;
}

DEFINE_validator(registry, &validateRegistry);

// Capture stdout to a string
class StdoutCapture {
 public:
  StdoutCapture() {
    oldCoutBuf_ = std::cout.rdbuf();
    std::cout.rdbuf(buffer_.rdbuf());
  }

  ~StdoutCapture() {
    std::cout.rdbuf(oldCoutBuf_);
  }

  std::string str() const {
    return buffer_.str();
  }

 private:
  std::stringstream buffer_;
  std::streambuf* oldCoutBuf_;
};

// Helper functions for LocalRunnerService
namespace {
std::shared_ptr<memory::MemoryPool> makeRootPool(const std::string& queryId) {
  static std::atomic_uint64_t poolId{0};
  return memory::memoryManager()->addRootPool(
      fmt::format("{}_{}", queryId, poolId++));
}

std::vector<RowVectorPtr> readCursor(
    std::shared_ptr<runner::LocalRunner>& runner,
    memory::MemoryPool* pool) {
  // We'll check the result after tasks are deleted, so copy the result
  // vectors to 'pool' that has longer lifetime.
  std::vector<RowVectorPtr> result;
  while (auto rows = runner->next()) {
    auto rowVector =
        std::dynamic_pointer_cast<RowVector>(BaseVector::copy(*rows, pool));
    if (rowVector) {
      result.push_back(rowVector);
    }
  }
  return result;
}

std::shared_ptr<core::QueryCtx> makeQueryCtx(
    const std::string& queryId,
    memory::MemoryPool* rootPool,
    folly::Executor* executor) {
  std::unordered_map<std::string, std::string> config;
  std::unordered_map<std::string, std::string> hiveConfig;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs;
  connectorConfigs["test-hive"] =
      std::make_shared<config::ConfigBase>(std::move(hiveConfig));

  return core::QueryCtx::create(
      executor,
      core::QueryConfig(config),
      std::move(connectorConfigs),
      cache::AsyncDataCache::getInstance());
}

// Create a MultiFragmentPlan from a single PlanNode
runner::MultiFragmentPlanPtr createSingleFragmentPlan(
    const core::PlanNodePtr& plan,
    const std::string& queryId,
    int32_t numWorkers,
    int32_t numDrivers) {
  runner::MultiFragmentPlan::Options options = {
      .queryId = queryId, .numWorkers = numWorkers, .numDrivers = numDrivers};

  // Create a single fragment with the given plan
  runner::ExecutableFragment fragment{queryId};
  fragment.width = 1; // Single task
  fragment.fragment = core::PlanFragment{plan};

  // Get all table scan nodes from the plan
  std::vector<core::TableScanNodePtr> scans;
  std::function<void(const core::PlanNodePtr&)> collectScans =
      [&scans, &collectScans](const core::PlanNodePtr& node) {
        if (auto tableScan =
                std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
          scans.push_back(tableScan);
        }
        for (const auto& source : node->sources()) {
          collectScans(source);
        }
      };
  collectScans(plan);
  fragment.scans = std::move(scans);

  return std::make_shared<runner::MultiFragmentPlan>(
      std::vector<runner::ExecutableFragment>{fragment}, std::move(options));
}

// Create a SimpleSplitSourceFactory with empty splits
std::shared_ptr<runner::SimpleSplitSourceFactory>
createEmptySplitSourceFactory() {
  std::unordered_map<
      core::PlanNodeId,
      std::vector<std::shared_ptr<connector::ConnectorSplit>>>
      nodeSplitMap;
  return std::make_shared<runner::SimpleSplitSourceFactory>(
      std::move(nodeSplitMap));
}
} // namespace

// Handler for HTTP requests
class LocalRunnerHandler : public RequestHandler {
 public:
  void onRequest(std::unique_ptr<HTTPMessage> request) noexcept override {
    request_ = std::move(request);
  }

  void onBody(std::unique_ptr<folly::IOBuf> body) noexcept override {
    if (body_) {
      body_->prependChain(std::move(body));
    } else {
      body_ = std::move(body);
    }
  }

  void onEOM() noexcept override {
    try {
      // Parse the request body as JSON
      std::string bodyStr;
      if (body_) {
        bodyStr = body_->moveToFbString().toStdString();
      }

      folly::dynamic requestJson = folly::parseJson(bodyStr);

      // Extract serialized plan from the JSON
      std::string serializedPlan =
          requestJson.getDefault("serialized_plan", "").asString();
      std::string queryId =
          requestJson.getDefault("query_id", "query").asString();
      int numWorkers =
          requestJson.getDefault("num_workers", FLAGS_num_workers).asInt();
      int numDrivers =
          requestJson.getDefault("num_drivers", FLAGS_num_drivers).asInt();

      // Validate required parameters
      if (serializedPlan.empty()) {
        ResponseBuilder(downstream_)
            .status(400, "Bad Request")
            .body("serialized_plan is required")
            .sendWithEOM();
        return;
      }

      // Capture stdout to return as part of the response
      StdoutCapture stdoutCapture;

      // Create memory pools
      std::shared_ptr<memory::MemoryPool> rootPool = makeRootPool(queryId);
      std::shared_ptr<memory::MemoryPool> pool =
          memory::memoryManager()->addLeafPool("output");

      // Deserialize the plan
      core::PlanNodePtr plan;
      try {
        folly::dynamic planJson = folly::parseJson(serializedPlan);
        plan =
            core::PlanNode::deserialize<core::PlanNode>(planJson, pool.get());
      } catch (const std::exception& e) {
        ResponseBuilder(downstream_)
            .status(400, "Bad Request")
            .body(fmt::format("Failed to deserialize plan: {}", e.what()))
            .sendWithEOM();
        return;
      }

      // Create a MultiFragmentPlan from the deserialized plan
      auto multiFragmentPlan =
          createSingleFragmentPlan(plan, queryId, numWorkers, numDrivers);

      // Create a LocalRunner
      executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(4);
      auto localRunner = std::make_shared<runner::LocalRunner>(
          multiFragmentPlan,
          makeQueryCtx(queryId, rootPool.get(), executor_.get()),
          createEmptySplitSourceFactory());

      // Run the query and collect results
      std::vector<RowVectorPtr> results;
      try {
        results = readCursor(localRunner, pool.get());
        localRunner->waitForCompletion(500'000); // 500ms timeout
      } catch (const std::exception& e) {
        ResponseBuilder(downstream_)
            .status(500, "Internal Server Error")
            .body(fmt::format("Error executing query: {}", e.what()))
            .sendWithEOM();
        return;
      }

      // Convert results to JSON
      folly::dynamic resultsJson = folly::dynamic::array;
      for (const auto& rowVector : results) {
        if (!rowVector) {
          continue; // Skip null vectors
        }
        folly::dynamic rowsJson = folly::dynamic::array;
        for (vector_size_t i = 0; i < rowVector->size(); ++i) {
          folly::dynamic rowJson = folly::dynamic::object;
          for (auto colIdx = 0; colIdx < rowVector->childrenSize(); ++colIdx) {
            if (!rowVector->type() || !rowVector->type()->isRow()) {
              continue; // Skip if type is null or not a row type
            }
            const auto& name = rowVector->type()->asRow().nameOf(colIdx);
            const auto& vector = rowVector->childAt(colIdx);

            if (vector->isNullAt(i)) {
              rowJson[name] = nullptr;
            } else {
              // Handle different vector types
              switch (vector->typeKind()) {
                case TypeKind::BOOLEAN:
                  rowJson[name] = vector->as<SimpleVector<bool>>()->valueAt(i);
                  break;
                case TypeKind::TINYINT:
                  rowJson[name] =
                      vector->as<SimpleVector<int8_t>>()->valueAt(i);
                  break;
                case TypeKind::SMALLINT:
                  rowJson[name] =
                      vector->as<SimpleVector<int16_t>>()->valueAt(i);
                  break;
                case TypeKind::INTEGER:
                  rowJson[name] =
                      vector->as<SimpleVector<int32_t>>()->valueAt(i);
                  break;
                case TypeKind::BIGINT:
                  rowJson[name] =
                      vector->as<SimpleVector<int64_t>>()->valueAt(i);
                  break;
                case TypeKind::REAL:
                  rowJson[name] = vector->as<SimpleVector<float>>()->valueAt(i);
                  break;
                case TypeKind::DOUBLE:
                  rowJson[name] =
                      vector->as<SimpleVector<double>>()->valueAt(i);
                  break;
                case TypeKind::VARCHAR:
                  rowJson[name] =
                      vector->as<SimpleVector<StringView>>()->valueAt(i).str();
                  break;
                default:
                  rowJson[name] = fmt::format("[{}]", vector->toString(i));
              }
            }
          }
          rowsJson.push_back(std::move(rowJson));
        }
        resultsJson.push_back(std::move(rowsJson));
      }

      // Create response JSON
      folly::dynamic responseJson = folly::dynamic::object;
      responseJson["status"] = "success";
      responseJson["output"] = stdoutCapture.str();
      responseJson["results"] = std::move(resultsJson);

      // Send the response
      ResponseBuilder(downstream_)
          .status(200, "OK")
          .header("Content-Type", "application/json")
          .body(folly::toJson(responseJson))
          .sendWithEOM();

    } catch (const std::exception& e) {
      // Handle exceptions
      folly::dynamic errorJson = folly::dynamic::object;
      errorJson["status"] = "error";
      errorJson["message"] = e.what();

      ResponseBuilder(downstream_)
          .status(500, "Internal Server Error")
          .header("Content-Type", "application/json")
          .body(folly::toJson(errorJson))
          .sendWithEOM();
    }
  }

  void onUpgrade(UpgradeProtocol /*protocol*/) noexcept override {
    // Not implemented
  }

  void requestComplete() noexcept override {
    delete this;
  }

  void onError(ProxygenError /*err*/) noexcept override {
    delete this;
  }

 private:
  std::unique_ptr<HTTPMessage> request_;
  std::unique_ptr<folly::IOBuf> body_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
};

// Factory for creating request handlers
class LocalRunnerHandlerFactory : public RequestHandlerFactory {
 public:
  void onServerStart(folly::EventBase* /*evb*/) noexcept override {}

  void onServerStop() noexcept override {}

  RequestHandler* onRequest(RequestHandler*, HTTPMessage*) noexcept override {
    return new LocalRunnerHandler();
  }
};

int main(int argc, char** argv) {
  // Initialize gflags and glog
  folly::Init init(&argc, &argv);

  // Initialize memory manager
  memory::initializeMemoryManager(memory::MemoryManager::Options{});

  // Register file systems and connectors
  filesystems::registerLocalFileSystem();
  connector::registerConnectorFactory(
      std::make_shared<connector::hive::HiveConnectorFactory>());
  dwrf::registerDwrfWriterFactory();

  Type::registerSerDe();
  core::PlanNode::registerSerDe();
  core::ITypedExpr::registerSerDe();

  functions::prestosql::registerAllScalarFunctions("presto.default.");
  aggregate::prestosql::registerAllAggregateFunctions(
      "presto.default.", false, true);
  window::prestosql::registerAllWindowFunctions("presto.default.");

  // Set up the HTTP server
  std::vector<HTTPServer::IPConfig> IPs = {
      {folly::SocketAddress("127.0.0.1", FLAGS_port),
       HTTPServer::Protocol::HTTP}};

  HTTPServerOptions options;
  options.threads = 4;
  options.idleTimeout = std::chrono::milliseconds(60000);
  options.shutdownOn = {SIGINT, SIGTERM};
  options.enableContentCompression = false;
  options.handlerFactories =
      RequestHandlerChain().addThen<LocalRunnerHandlerFactory>().build();

  // Create and start the server
  HTTPServer server(std::move(options));
  server.bind(IPs);

  LOG(INFO) << "Starting LocalRunner service on port " << FLAGS_port;

  // Start the server
  std::thread t([&]() { server.start(); });

  // Wait for the server to exit
  t.join();

  return 0;
}
