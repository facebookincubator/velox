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

#include <boost/asio.hpp>
#include <folly/init/Init.h>
#include "RemoteFunctionRestService.h"
#include "velox/common/memory/Memory.h"
#include "velox/exec/tests/utils/PortUtil.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

/// This executable is meant for testing. It instantiates a lightweight
/// server that can handle remote function requests, hosting all Presto scalar
/// functions. Clients can connect to this server to invoke these functions
/// remotely for testing and validation purposes.
///
/// The server binds to a TCP endpoint specified by the --service_host and
/// --service_port flags, and each function is registered with a prefix defined
/// by the --function_prefix flag.
///
/// NOTE: This server runs on a single-threaded boost::asio::io_context for
/// simplicity; it is not optimized for high throughput or production use.

DEFINE_string(service_host, "127.0.0.1", "Host to bind the service to");
DEFINE_string(
    function_prefix,
    "remote.schema",
    "Prefix to be added to the functions being registered");

using namespace ::facebook::velox;

int main(int argc, char* argv[]) {
  folly::Init init(&argc, &argv);
  FLAGS_logtostderr = true;
  memory::initializeMemoryManager({});

  LOG(INFO) << "Registering Presto functions";
  functions::prestosql::registerAllScalarFunctions(FLAGS_function_prefix);
  boost::asio::io_context ioc{1};

  auto servicePort = facebook::velox::exec::test::getFreePort();
  LOG(INFO) << "Initializing rest server at 127.0.0.1:" << servicePort;
  std::make_shared<functions::RestListener>(
      ioc,
      boost::asio::ip::tcp::endpoint(
          boost::asio::ip::make_address(FLAGS_service_host), servicePort),
      FLAGS_function_prefix)
      ->run();

  ioc.run();

  return 0;
}
