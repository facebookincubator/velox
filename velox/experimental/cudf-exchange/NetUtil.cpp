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
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <cstring>
#include <set>
#include <string>

// Helper: resolve a hostname/IP string into a set of stringified addresses
static std::set<std::string> resolveHost(const std::string& host) {
  std::set<std::string> results;
  addrinfo hints{}, *res, *p;

  hints.ai_family = AF_UNSPEC; // Allow both IPv4 and IPv6
  hints.ai_socktype = SOCK_STREAM; // Any type is fine
  hints.ai_flags = AI_ADDRCONFIG; // Use configured address families

  int status = getaddrinfo(host.c_str(), nullptr, &hints, &res);
  if (status != 0) {
    // Could not resolve
    return results;
  }

  for (p = res; p != nullptr; p = p->ai_next) {
    char ipstr[INET6_ADDRSTRLEN];
    void* addr = nullptr;

    if (p->ai_family == AF_INET) { // IPv4
      sockaddr_in* ipv4 = reinterpret_cast<sockaddr_in*>(p->ai_addr);
      addr = &(ipv4->sin_addr);
      inet_ntop(AF_INET, addr, ipstr, sizeof(ipstr));
      results.insert(ipstr);
    } else if (p->ai_family == AF_INET6) { // IPv6
      sockaddr_in6* ipv6 = reinterpret_cast<sockaddr_in6*>(p->ai_addr);
      addr = &(ipv6->sin6_addr);
      inet_ntop(AF_INET6, addr, ipstr, sizeof(ipstr));
      results.insert(ipstr);
    }
  }

  freeaddrinfo(res);
  return results;
}

// Main function: check if two hosts resolve to a common address
bool isSameHost(const std::string& h1, const std::string& h2) {
  auto set1 = resolveHost(h1);
  auto set2 = resolveHost(h2);

  for (const auto& addr : set1) {
    if (set2.find(addr) != set2.end()) {
      return true;
    }
  }
  return false;
}
