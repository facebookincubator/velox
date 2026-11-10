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
#include "velox/dwio/nimble/reader/NimbleReaderFactory.h"

#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox::nimble {

std::unique_ptr<dwio::common::Reader> NimbleReaderFactory::createReader(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options) {
  // Missing metadata IO stats are defaulted by
  // `TabletReader::configureOptions`, which both branches below funnel
  // through, so no fixup is needed here.
  if (options.selectiveNimbleReaderEnabled()) {
    addThreadLocalRuntimeStat("selectiveNimbleReader", RuntimeCounter(1));
    return selectiveFactory_.createReader(std::move(input), options);
  }

  addThreadLocalRuntimeStat("batchNimbleReader", RuntimeCounter(1));
  return detail::createBatchReader(
      options, input->getReadFile(), input->getInputStream()->fileIoContext());
}

void registerNimbleReaderFactory() {
  dwio::common::registerReaderFactory(std::make_shared<NimbleReaderFactory>());
}

void unregisterNimbleReaderFactory() {
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::NIMBLE);
}

} // namespace facebook::velox::nimble
