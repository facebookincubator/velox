/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/common/MetricsLogger.h"

namespace facebook::nimble {

folly::dynamic StripeLoadMetrics::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["stripeIndex"] = stripeIndex;
  obj["rowsInStripe"] = rowsInStripe;
  obj["streamCount"] = streamCount;
  obj["totalStreamSize"] = totalStreamSize;
  return obj;
}

folly::dynamic StripeFlushMetrics::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["inputSize"] = inputSize;
  obj["rowCount"] = rowCount;
  obj["stripeSize"] = stripeSize;
  obj["trackedMemory"] = trackedMemory;
  return obj;
}

folly::dynamic FileCloseMetrics::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["rowCount"] = rowCount;
  obj["inputSize"] = inputSize;
  obj["stripeCount"] = stripeCount;
  obj["fileSize"] = fileSize;
  obj["encodingCpuNs"] = encodingCpuNs;
  obj["encodingWallNs"] = encodingWallNs;
  return obj;
}

LoggingScope::Context& LoggingScope::Context::get() {
  thread_local static Context ctx;
  return ctx;
}

} // namespace facebook::nimble
