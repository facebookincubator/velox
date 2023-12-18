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

#include "velox/common/file/tests/BlockingReadFile.h"

namespace facebook::velox::tests::utils {

bool BlockingReadFile::signalPreadv() {
  auto batons = batons_.wlock();
  if (batons->empty()) {
    return false;
  }
  batons->front()->post();
  batons->pop();
  return true;
}

void BlockingReadFile::setNumberOfReadNextWithoutBlocking(
    uint32_t numberOfReads) {
  VELOX_CHECK(
      batons_.wlock()->empty(),
      "Called BlockingReadFile#setNumberOfReadNextWithoutBlocking while a read was in progress. Ensure queue is empty before calling");
  readsToSkipBlocking_.withWLock(
      [numberOfReads](auto& reads) { reads = numberOfReads; });
}

void BlockingReadFile::forceBlockOnNextRead() {
  readsToSkipBlocking_.withWLock([](auto& reads) { reads = 0; });
}

void BlockingReadFile::preadv(
    folly::Range<const velox::common::Region*> regions,
    folly::Range<folly::IOBuf*> iobufs) const {
  waitOnReadCondition();
  file_->preadv(regions, iobufs);
}

uint64_t BlockingReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) const {
  waitOnReadCondition();
  return file_->preadv(offset, buffers);
}

void BlockingReadFile::waitOnReadCondition() const {
  readsToSkipBlocking_.withWLock([](auto& reads) {
    LOG(INFO) << "BlockingReadFile::waitOnReadCondition with " << reads;
    if (reads > 0) {
      reads--;
    }
  });
  std::shared_ptr<folly::Baton<>> baton = std::make_shared<folly::Baton<>>();
  batons_.wlock()->push(baton);
  baton->wait();
}

} // namespace facebook::velox::tests::utils
