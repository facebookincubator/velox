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

#include "velox/common/testutil/PageBeforeGuard.h"

#include <sys/mman.h>
#include <unistd.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::testutil {

PageBeforeGuard::PageBeforeGuard() : pageSize_(::sysconf(_SC_PAGESIZE)) {
  void* mapping = ::mmap(
      nullptr,
      2 * pageSize_,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
  VELOX_CHECK_NE(mapping, MAP_FAILED);
  begin_ = static_cast<uint8_t*>(mapping);
  if (::mprotect(end(), pageSize_, PROT_NONE) != 0) {
    ::munmap(mapping, 2 * pageSize_);
    VELOX_FAIL("Failed to protect the guard page.");
  }
}

PageBeforeGuard::~PageBeforeGuard() {
  ::munmap(begin_, 2 * pageSize_);
}

} // namespace facebook::velox::common::testutil
