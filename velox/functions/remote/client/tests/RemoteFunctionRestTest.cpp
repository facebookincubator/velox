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

#include <cstdio>

#include <boost/asio.hpp>
#include <folly/SocketAddress.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/PortUtil.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/remote/client/Remote.h"
#include "velox/functions/remote/utils/restserver/RemoteFunctionRestService.h"

using ::facebook::velox::test::assertEqualVectors;

namespace facebook::velox::functions {
namespace {

class RemoteFunctionRestTest
    : public test::FunctionBaseTest,
      public testing::WithParamInterface<remote::PageFormat> {
 public:
  void SetUp() override {
    auto servicePort = exec::test::getFreePort();
    location_ = fmt::format("http://127.0.0.1:{}", servicePort);
    auto wrongServicePort = exec::test::getFreePort();
    wrongLocation_ = fmt::format("http://127.0.0.1:{}", wrongServicePort);

    initializeServer(servicePort);
    registerRemoteFunctions();
  }

  ~RemoteFunctionRestTest() override {
    if (serverThread_ && serverThread_->joinable()) {
      ioc_.stop();
      serverThread_->join();
    }
  }

 private:
  // Registers a remote functions to be used in this test.
  void registerRemoteFunctions() const {
    auto absSignature = {exec::FunctionSignatureBuilder()
                             .returnType("integer")
                             .argumentType("integer")
                             .build()};
    auto strLenSignature = {exec::FunctionSignatureBuilder()
                                .returnType("integer")
                                .argumentType("varchar")
                                .build()};
    auto strTrimSignature = {exec::FunctionSignatureBuilder()
                                 .returnType("varchar")
                                 .argumentType("varchar")
                                 .build()};
    auto divSignatures = {exec::FunctionSignatureBuilder()
                              .returnType("double")
                              .argumentType("double")
                              .argumentType("double")
                              .build()};

    auto registerFunction =
        [&](const std::string& functionName,
            const std::vector<exec::FunctionSignaturePtr>& signatures,
            const std::string& baseLocation) {
          RemoteVectorFunctionMetadata metadata;
          metadata.serdeFormat = GetParam();
          metadata.location = baseLocation + '/' + functionName;
          registerRemoteFunction(functionName, signatures, metadata);
        };

    registerFunction("remote_abs", absSignature, location_);
    registerFunction("remote_strlen", strLenSignature, location_);
    registerFunction("remote_trim", strTrimSignature, location_);
    registerFunction("remote_divide", divSignatures, location_);
    registerFunction("remote_round", absSignature, location_);
    registerFunction("remote_wrong_port", absSignature, wrongLocation_);
  }

  void initializeServer(uint16_t servicePort) {
    // Adjusted for Boost.Beast server; the server is started in the main
    // thread.

    // Start the server in a separate thread
    serverThread_ = std::make_unique<std::thread>([this, servicePort]() {
      std::string serviceHost = "127.0.0.1";
      std::make_shared<RestListener>(
          ioc_,
          boost::asio::ip::tcp::endpoint(
              boost::asio::ip::make_address(serviceHost), servicePort))
          ->run();

      ioc_.run();
    });

    VELOX_CHECK(
        waitForRunning(servicePort), "Unable to initialize HTTP server.");
  }

  bool waitForRunning(uint16_t servicePort) const {
    for (size_t i = 0; i < 100; ++i) {
      using boost::asio::ip::tcp;
      boost::asio::io_context io_context;

      tcp::socket socket(io_context);
      tcp::resolver resolver(io_context);

      try {
        boost::asio::connect(
            socket, resolver.resolve("127.0.0.1", std::to_string(servicePort)));
        return true;
      } catch (std::exception& e) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }
    return false;
  }

  std::unique_ptr<std::thread> serverThread_;
  boost::asio::io_context ioc_{1};

  std::string location_;
  std::string wrongLocation_;
};

TEST_P(RemoteFunctionRestTest, absolute) {
  auto inputVector = makeFlatVector<int32_t>({-10, -20});
  auto results = evaluate<SimpleVector<int32_t>>(
      "remote_abs(c0)", makeRowVector({inputVector}));

  auto expected = makeFlatVector<int32_t>({10, 20});
  assertEqualVectors(expected, results);
}

TEST_P(RemoteFunctionRestTest, stringLength) {
  auto inputVector =
      makeFlatVector<StringView>({"hello", "from", "remote", "server"});
  auto results = evaluate<SimpleVector<int32_t>>(
      "remote_strlen(c0)", makeRowVector({inputVector}));

  auto expected = makeFlatVector<int32_t>({5, 4, 6, 6});
  assertEqualVectors(expected, results);
}

TEST_P(RemoteFunctionRestTest, trimWhitespace) {
  auto inputVector = makeFlatVector<StringView>(
      {"hello from remote server", "testing remote server"});
  auto results = evaluate<SimpleVector<StringView>>(
      "remote_trim(c0)", makeRowVector({inputVector}));

  auto expected = makeFlatVector<StringView>(
      {"hellofromremoteserver", "testingremoteserver"});
  assertEqualVectors(expected, results);
}

TEST_P(RemoteFunctionRestTest, tryException) {
  // remote_divide throws if denominator is 0.
  auto numeratorVector = makeFlatVector<double>({0, 1, 4, 9, 16, 25, -25});
  auto denominatorVector = makeFlatVector<double>({0, 1, 2, 3, 4, 0, 2});
  auto data = makeRowVector({numeratorVector, denominatorVector});
  auto results = evaluate<SimpleVector<double>>("remote_divide(c0, c1)", data);

  ASSERT_EQ(results->size(), 7);
  auto expected = makeFlatVector<double>({0, 1, 2, 3, 4, 0, -12.5});
  expected->setNull(0, true);
  expected->setNull(5, true);

  assertEqualVectors(expected, results);
}

TEST_P(RemoteFunctionRestTest, functionNotAvailable) {
  auto inputVector = makeFlatVector<int32_t>({-10, -20});
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<int32_t>>(
          "remote_round(c0)", makeRowVector({inputVector})),
      "Function 'remote_round' is not available on the server.");
}

TEST_P(RemoteFunctionRestTest, connectionError) {
  auto inputVector = makeFlatVector<int32_t>({-10, -20});
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<int32_t>>(
          "remote_wrong_port(c0)", makeRowVector({inputVector})),
      "Error communicating with server: ");
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    RemoteFunctionRestTestFixture,
    RemoteFunctionRestTest,
    ::testing::Values(
        remote::PageFormat::PRESTO_PAGE,
        remote::PageFormat::SPARK_UNSAFE_ROW));

} // namespace
} // namespace facebook::velox::functions

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
