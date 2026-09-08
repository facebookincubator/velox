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

#include <numeric>

namespace facebook::velox {

namespace {

const raw_vector<int32_t> kIotaData = [] {
  raw_vector<int32_t> v;
  v.resize(1'000'000);
  v.resize(v.capacity());
  std::iota(v.begin(), v.end(), 0);
  return v;
}();

} // namespace

const int32_t*
iota(int32_t size, raw_vector<int32_t>& storage, int32_t offset) {
  if (kIotaData.size() < offset + size) {
    storage.resize(size);
    std::iota(storage.begin(), storage.end(), offset);
    return storage.data();
  }

  return kIotaData.data() + offset;
}

} // namespace facebook::velox
