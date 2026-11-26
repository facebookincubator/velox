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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "velox/experimental/ucx-exchange/Acceptor.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"

namespace facebook::velox::ucx_exchange {
namespace {

// Stands in for the UCP handle of an incoming endpoint. The Acceptor only
// compares handles and never dereferences them.
ucp_ep_h fakeHandle(uintptr_t value) {
  return reinterpret_cast<ucp_ep_h>(value);
}

// UCP can reuse the handle of an endpoint that closed for a new connection.
// Handshakes on the new connection must resolve to its endpoint, not to the
// closed one.
TEST(AcceptorTest, reusedHandleMapsToNewEndpoint) {
  Acceptor acceptor;
  auto closed = std::make_shared<EndpointRef>(nullptr);
  auto reconnected = std::make_shared<EndpointRef>(nullptr);
  acceptor.registerEndpointRef(fakeHandle(0x10), closed);
  acceptor.registerEndpointRef(fakeHandle(0x10), reconnected);
  EXPECT_EQ(acceptor.findEndpointRef(fakeHandle(0x10)), reconnected);
}

// Removing an endpoint erases its handle, so that the map does not keep an
// entry per connection for the life of the worker.
TEST(AcceptorTest, unregisterErasesHandle) {
  Acceptor acceptor;
  auto removed = std::make_shared<EndpointRef>(nullptr);
  auto other = std::make_shared<EndpointRef>(nullptr);
  acceptor.registerEndpointRef(fakeHandle(0x10), removed);
  acceptor.registerEndpointRef(fakeHandle(0x20), other);
  acceptor.unregisterEndpointRef(removed);
  EXPECT_EQ(acceptor.findEndpointRef(fakeHandle(0x10)), nullptr);
  EXPECT_EQ(acceptor.findEndpointRef(fakeHandle(0x20)), other);
}

// An endpoint removed after a new connection reused its handle leaves the
// mapping of the new connection in place.
TEST(AcceptorTest, unregisterKeepsMappingOfReusedHandle) {
  Acceptor acceptor;
  auto closed = std::make_shared<EndpointRef>(nullptr);
  auto reconnected = std::make_shared<EndpointRef>(nullptr);
  acceptor.registerEndpointRef(fakeHandle(0x10), closed);
  acceptor.registerEndpointRef(fakeHandle(0x10), reconnected);
  acceptor.unregisterEndpointRef(closed);
  EXPECT_EQ(acceptor.findEndpointRef(fakeHandle(0x10)), reconnected);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
