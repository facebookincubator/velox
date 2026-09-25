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

#include "velox/connectors/hive/storage_adapters/abfs/AsyncChannelFactory.h"

namespace folly {
class EventBase;
}

namespace facebook::velox::filesystems {

/// Creates plaintext AsyncSocket channels on one owning EventBase.
class EventSocketChannelFactory final : public AsyncChannelFactory {
 public:
  /// Creates a factory using the supplied non-null owning EventBase.
  explicit EventSocketChannelFactory(folly::EventBase* eventBase);

  /// Creates a factory using the supplied owning EventBase.
  explicit EventSocketChannelFactory(folly::EventBase& eventBase);

  /// Connects to the endpoint's pre-resolved address from a fiber.
  folly::AsyncTransportWrapper::UniquePtr connect(
      const AsyncChannelEndpoint& endpoint) override;

 private:
  folly::EventBase* eventBase_;
};

} // namespace facebook::velox::filesystems
