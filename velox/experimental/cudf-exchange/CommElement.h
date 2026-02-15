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

#include <stdexcept>

// The CommElement is the abstract base class of both the
// per-client context on the exchange server side as well as the
// exchange source side.
namespace facebook::velox::cudf_exchange {

class Communicator;
class EndpointRef;

class CommElement {
 public:
  CommElement(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef)
      : communicator_{communicator}, endpointRef_{endpointRef} {}

  CommElement(const std::shared_ptr<Communicator> communicator)
      : communicator_{communicator}, endpointRef_{nullptr} {}

  virtual ~CommElement() = default;

  /// @brief Advance the communication by executing the communication elements
  /// specific communication pattern.
  virtual void process() = 0;

  // Called when the underlying endpoint was closed
  // or the communicator is finished.
  virtual void close() = 0;

 protected:
  const std::shared_ptr<Communicator> communicator_;
  std::shared_ptr<EndpointRef> endpointRef_;
};
} // namespace facebook::velox::cudf_exchange
