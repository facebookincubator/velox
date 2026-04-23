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
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>

#include <glog/logging.h>

// Helper: resolve a hostname/IP string into a set of stringified addresses
static std::set<std::string> resolveHost(std::string_view host) {
  std::set<std::string> results;
  addrinfo hints{}, *res, *p;

  hints.ai_family = AF_UNSPEC; // Allow both IPv4 and IPv6
  hints.ai_socktype = SOCK_STREAM; // Any type is fine
  hints.ai_flags = AI_ADDRCONFIG; // Use configured address families

  std::string hostStr{host};
  int status = getaddrinfo(hostStr.c_str(), nullptr, &hints, &res);
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
bool isSameHost(std::string_view h1, std::string_view h2) {
  auto set1 = resolveHost(h1);
  auto set2 = resolveHost(h2);

  for (const auto& addr : set1) {
    if (set2.find(addr) != set2.end()) {
      return true;
    }
  }
  return false;
}

// Get all IP addresses assigned to local network interfaces.
// Used for server-side intra-node detection.
std::unordered_set<std::string> getLocalIpAddresses() {
  std::unordered_set<std::string> localIps;
  struct ifaddrs* ifaddr = nullptr;
  struct ifaddrs* ifa = nullptr;

  if (getifaddrs(&ifaddr) == -1) {
    LOG(WARNING) << "[EXCHANGE_DEBUG] getLocalIpAddresses: getifaddrs() failed";
    return localIps;
  }

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) {
      continue;
    }

    char ipstr[INET6_ADDRSTRLEN];
    if (ifa->ifa_addr->sa_family == AF_INET) {
      // IPv4
      sockaddr_in* ipv4 = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
      inet_ntop(AF_INET, &ipv4->sin_addr, ipstr, sizeof(ipstr));
      localIps.insert(ipstr);
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {
      // IPv6
      sockaddr_in6* ipv6 = reinterpret_cast<sockaddr_in6*>(ifa->ifa_addr);
      inet_ntop(AF_INET6, &ipv6->sin6_addr, ipstr, sizeof(ipstr));
      localIps.insert(ipstr);
    }
  }

  // CRITICAL: Use freeifaddrs(), not freeaddrinfo()
  freeifaddrs(ifaddr);

  // Log all local IPs for debugging
  std::ostringstream oss;
  oss << "{";
  bool first = true;
  for (const auto& ip : localIps) {
    if (!first)
      oss << ", ";
    oss << ip;
    first = false;
  }
  oss << "}";
  LOG(INFO) << "[EXCHANGE_DEBUG] getLocalIpAddresses: " << localIps.size()
            << " addresses: " << oss.str();

  return localIps;
}
