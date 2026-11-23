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

#include "velox/common/memory/RawVector.h"

#include <algorithm>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_int32(
    velox_shared_iota_size,
    1'000'000,
    "Number of consecutive values that iota() serves from a process-wide "
    "table. Larger requests are written to the caller's storage. Read once, on "
    "the first iota() call.");

namespace facebook::velox {

namespace {

// Built on first use so that FLAGS_velox_shared_iota_size can be set at
// startup.
const raw_vector<int32_t>& iotaData() {
  static const raw_vector<int32_t> data = [] {
    raw_vector<int32_t> v;
    v.resize(std::max(FLAGS_velox_shared_iota_size, 0));
    v.resize(v.capacity());
    std::iota(v.begin(), v.end(), 0);
    LOG(INFO) << "Built shared iota table with " << v.size() << " values";
    return v;
  }();
  return data;
}

} // namespace

const int32_t*
iota(int32_t size, raw_vector<int32_t>& storage, int32_t offset) {
  const auto& data = iotaData();
  if (data.size() < static_cast<int64_t>(offset) + size) {
    storage.resize(size);
    std::iota(storage.begin(), storage.end(), offset);
    return storage.data();
  }

  return data.data() + offset;
}

} // namespace facebook::velox
