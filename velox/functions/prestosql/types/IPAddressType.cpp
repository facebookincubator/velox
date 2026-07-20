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

#include "velox/functions/prestosql/types/IPAddressType.h"

namespace facebook::velox {

std::string IPAddressType::valueToString(int128_t value) const {
  auto bytes = ipaddress::toIPv6ByteArray(value);
  folly::IPAddressV6 v6Addr(bytes);
  if (v6Addr.isIPv4Mapped()) {
    return v6Addr.createIPv4().str();
  }
  return v6Addr.str();
}

} // namespace facebook::velox
