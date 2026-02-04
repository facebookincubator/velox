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

#include "velox/functions/remote/client/Remote.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <random>
#include <thread>

#include <folly/io/async/AsyncSocketException.h>
#include <folly/io/async/EventBase.h>
#include <thrift/lib/cpp/transport/TTransportException.h>
#include "velox/functions/remote/client/RemoteVectorFunction.h"
#include "velox/functions/remote/client/ThriftClient.h"
#include "velox/functions/remote/if/GetSerde.h"
#include "velox/functions/remote/if/gen-cpp2/RemoteFunctionServiceAsyncClient.h"

DEFINE_int32(
    remote_function_retry_count,
    3,
    "Number of retries for remote function calls on transport errors");

DEFINE_int32(
    remote_function_retry_max_backoff_sec,
    8,
    "Maximum exponential backoff in seconds for remote function retries");

namespace facebook::velox::functions {
namespace {

class RemoteThriftFunction : public RemoteVectorFunction {
 public:
  RemoteThriftFunction(
      const std::string& functionName,
      const std::vector<exec::VectorFunctionArg>& inputArgs,
      const RemoteThriftVectorFunctionMetadata& metadata)
      : RemoteVectorFunction(functionName, inputArgs, metadata),
        functionName_(functionName),
        location_(metadata.location),
        client_(createClient(metadata)) {
    VLOG(1) << "Created RemoteThriftFunction '" << functionName_ << "' for "
            << location_.describe();
  }

  std::unique_ptr<remote::RemoteFunctionResponse> invokeRemoteFunction(
      const remote::RemoteFunctionRequest& request) const override {
    auto remoteResponse = std::make_unique<remote::RemoteFunctionResponse>();

    int retryCount = 0;
    int expIntervalSec = 1;

    while (true) {
      try {
        VLOG(2) << "Invoking remote function '" << functionName_
                << "' (socket=" << location_.describe() << ")";

        client_->invokeFunction(*remoteResponse, request);

        VLOG(2) << "Remote function '" << functionName_ << "' call succeeded";
        return remoteResponse;

      } catch (const apache::thrift::transport::TTransportException& e) {
        if (!handleRetryableError(e.what(), retryCount, expIntervalSec)) {
          throw;
        }
      } catch (const folly::AsyncSocketException& e) {
        std::string errorMsg = fmt::format(
            "{} (type={})", e.what(), static_cast<int>(e.getType()));
        if (!handleRetryableError(errorMsg, retryCount, expIntervalSec)) {
          throw;
        }
      }
    }
  }

  std::string remoteLocationToString() const override {
    return location_.describe();
  }

 private:
  std::unique_ptr<IRemoteFunctionClient> createClient(
      const RemoteThriftVectorFunctionMetadata& metadata) {
    if (metadata.clientFactory) {
      clientFactory_ = metadata.clientFactory;
      return clientFactory_(metadata.location, &eventBase_);
    }
    clientFactory_ = getDefaultRemoteFunctionClient;
    return clientFactory_(metadata.location, &eventBase_);
  }

  // Handles retryable errors with exponential backoff.
  // Returns true if retry should continue, false if retries exhausted.
  bool handleRetryableError(
      const std::string& errorMsg,
      int& retryCount,
      int& expIntervalSec) const {
    LOG(ERROR) << "Transport error in remote function '" << functionName_
               << "': " << errorMsg << " (attempt=" << (retryCount + 1) << "/"
               << (FLAGS_remote_function_retry_count + 1) << ")";

    if (retryCount < FLAGS_remote_function_retry_count) {
      reconnectClient();
      sleepWithJitter(expIntervalSec);
      expIntervalSec = std::min(
          expIntervalSec * 2, FLAGS_remote_function_retry_max_backoff_sec);
      ++retryCount;
      return true;
    }

    LOG(ERROR) << "Remote function '" << functionName_ << "' call failed after "
               << FLAGS_remote_function_retry_count << " retries";
    return false;
  }

  void reconnectClient() const {
    LOG(WARNING) << "Reconnecting thrift client for '" << functionName_
                 << "' to " << location_.describe();
    client_ = clientFactory_(location_, &eventBase_);
  }

  void sleepWithJitter(int expIntervalSec) const {
    static thread_local std::mt19937 rng(std::random_device{}());
    // Use range [0.5, expIntervalSec + 0.5) to ensure meaningful backoff
    std::uniform_real_distribution<double> dist(0.5, expIntervalSec + 0.5);
    auto sleepIntervalSec = static_cast<long>(dist(rng));

    LOG(INFO) << "Sleeping for " << sleepIntervalSec
              << " seconds before retry for '" << functionName_ << "'";
    /* sleep override: intentional backoff for retry logic */
    std::this_thread::sleep_for(std::chrono::seconds(sleepIntervalSec));
  }

  const std::string functionName_;
  folly::SocketAddress location_;
  mutable folly::EventBase eventBase_;
  mutable RemoteFunctionClientFactory clientFactory_;
  mutable std::unique_ptr<IRemoteFunctionClient> client_;
};

std::shared_ptr<exec::VectorFunction> createRemoteFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/,
    const RemoteThriftVectorFunctionMetadata& metadata) {
  return std::make_unique<RemoteThriftFunction>(name, inputArgs, metadata);
}

} // namespace

void registerRemoteFunction(
    const std::string& name,
    std::vector<exec::FunctionSignaturePtr> signatures,
    const RemoteThriftVectorFunctionMetadata& metadata,
    bool overwrite) {
  exec::registerStatefulVectorFunction(
      name,
      signatures,
      std::bind(
          createRemoteFunction,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3,
          metadata),
      metadata,
      overwrite);
}

} // namespace facebook::velox::functions
