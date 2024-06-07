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
#include <proxygen/httpserver/HTTPServer.h>
#include "velox/common/memory/Memory.h"

#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/remote/server/RemoteFunctionRestService.h"

DEFINE_string(
    service_host,
    "127.0.0.1",
    "Prefix to be added to the functions being registered");

DEFINE_int32(
    service_port,
    8321,
    "Prefix to be added to the functions being registered");

DEFINE_string(
    function_prefix,
    "remote.schema.",
    "Prefix to be added to the functions being registered");

using namespace ::facebook::velox;

int main(int argc, char* argv[]) {
  folly::Init init(&argc, &argv);
  FLAGS_logtostderr = true;
  memory::initializeMemoryManager({});

  // A remote function service should handle the function execution by its own.
  // But we use Velox framework for quick prototype here
  functions::prestosql::registerAllScalarFunctions(FLAGS_function_prefix);
  //  registerFunction<functions::PlusFunction, int64_t, int64_t, int64_t>(
  //          {"remote_plus"});
  // End of function registration

  LOG(INFO) << "Start HTTP Server at " << "http://" << FLAGS_service_host << ":"
            << FLAGS_service_port;

  HTTPServerOptions options;
  //  options.threads = static_cast<size_t>(sysconf(_SC_NPROCESSORS_ONLN));
  options.idleTimeout = std::chrono::milliseconds(60000);
  options.handlerFactories =
      RequestHandlerChain()
          .addThen<functions::RestRequestHandlerFactory>()
          .build();
  options.h2cEnabled = true;

  std::vector<HTTPServer::IPConfig> IPs = {
      {folly::SocketAddress(FLAGS_service_host, FLAGS_service_port, true),
       HTTPServer::Protocol::HTTP}};

  proxygen::HTTPServer server(std::move(options));
  server.bind(IPs);

  std::thread t([&]() { server.start(); });

  t.join();
  return 0;
}
