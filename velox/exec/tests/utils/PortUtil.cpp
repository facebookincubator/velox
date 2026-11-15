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

#include "velox/exec/tests/utils/PortUtil.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <memory>
#include <thread>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::test {
namespace {

void getFreePortsImpl(int numPorts, int* ports) {
  const std::unique_ptr<int[]> sockets(new int[numPorts]);
  for (int i = 0; i < numPorts; i++) {
    int sock = socket(PF_INET, SOCK_STREAM, 0);
    VELOX_CHECK_NE(sock, -1, "Error while creating socket: {}", errno);
    sockets[i] = sock;

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = 0;
    addr.sin_addr.s_addr = INADDR_ANY;

    socklen_t len = sizeof(addr);
    int result = ::bind(sock, reinterpret_cast<sockaddr*>(&addr), len);
    VELOX_CHECK_NE(result, -1, "Error while binding socket: {}", errno);

    result = getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &len);
    VELOX_CHECK_NE(result, -1, "Error on getsockname: {}", errno);

    ports[i] = ntohs(addr.sin_port);
  }

  for (int i = 0; i < numPorts; i++) {
    close(sockets[i]);
  }
}

bool isPortReady(int port) {
  const int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  const bool ready =
      ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
  close(sock);
  return ready;
}

} // namespace

std::vector<int> PortUtil::getFreePorts(std::size_t numPorts) {
  std::vector<int> ports;
  ports.resize(numPorts);
  getFreePortsImpl(numPorts, &ports[0]);
  return ports;
}

int PortUtil::getFreePort() {
  int port;
  getFreePortsImpl(1, &port);
  return port;
}

void PortUtil::waitForPort(
    int port,
    std::string_view serverName,
    const std::function<bool()>& isRunning,
    std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (!isRunning()) {
      VELOX_FAIL("{} exited before listening on port {}", serverName, port);
    }
    if (isPortReady(port)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  VELOX_FAIL(
      "Timed out after {} seconds waiting for {} to listen on port {}",
      timeout.count(),
      serverName,
      port);
}

} // namespace facebook::velox::exec::test
