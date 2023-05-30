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

#include "velox/common/file/Utils.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::file::utils {

bool CoalesceIfDistanceLE::operator()(
    const ReadFile::Segment& a,
    const ReadFile::Segment& b) const {
  if (a.offset == b.offset && a.buffer.size() == b.buffer.size()) {
    // Support duplicate segments
    return true;
  }

  VELOX_CHECK_LE(
      a.offset,
      b.offset,
      "Segments must be sorted: a({}, {}, &{}, \"{}\"), b({}, {}, &{}, \"{}\")",
      a.offset,
      a.buffer.size(),
      static_cast<void*>(a.buffer.data()),
      a.label,
      b.offset,
      b.buffer.size(),
      static_cast<void*>(b.buffer.data()),
      b.label);
  const uint64_t beginGap = a.offset + a.buffer.size(), endGap = b.offset;

  VELOX_CHECK_LE(
      beginGap,
      endGap,
      "Segment overlap is not supported: a({}, {}, &{}, \"{}\"), b({}, {}, &{}, \"{}\")",
      a.offset,
      a.buffer.size(),
      static_cast<void*>(a.buffer.data()),
      a.label,
      b.offset,
      b.buffer.size(),
      static_cast<void*>(b.buffer.data()),
      b.label);
  const uint64_t gap = endGap - beginGap;

  return gap <= maxCoalescingDistance_;
}

} // namespace facebook::velox::file::utils
