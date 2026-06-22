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
#pragma once

#include <string>
#include <string_view>
#include <unordered_set>

/// @brief Utility function that checks whether two
/// hostnames resolve to the same underlying host.
// The hostnames can be domain names, IPv4 or IPv6 addresses.
bool isSameHost(std::string_view h1, std::string_view h2);

/// @brief Get all IP addresses assigned to local network interfaces.
/// Used for server-side intra-node detection by checking if a peer's IP
/// is in this set. Includes both IPv4 and IPv6 addresses, as well as
/// loopback addresses (127.0.0.1, ::1).
/// @return Set of IP address strings for all local interfaces.
std::unordered_set<std::string> getLocalIpAddresses();
