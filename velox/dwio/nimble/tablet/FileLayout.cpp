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

#include "velox/dwio/nimble/tablet/FileLayout.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/Options.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"

#include <numeric>

namespace facebook::nimble {

FileLayout FileLayout::create(
    std::shared_ptr<velox::ReadFile> file,
    velox::memory::MemoryPool* pool) {
  auto dataStats = std::make_shared<velox::io::IoStatistics>();
  auto metadataStats = std::make_shared<velox::io::IoStatistics>();
  auto indexStats = std::make_shared<velox::io::IoStatistics>();
  velox::io::ReaderOptions readerOptions(pool);
  readerOptions.setDataIoStats(dataStats);
  readerOptions.setMetadataIoStats(metadataStats);
  readerOptions.setIndexIoStats(indexStats);
  TabletReader::Options options;
  options.loadClusterIndex = true;
  options.ioOptions = readerOptions;
  auto tablet = TabletReader::create(std::move(file), pool, options);

  FileLayout layout;
  layout.fileSize = tablet->fileSize();

  // Postscript and footer
  layout.postscript = tablet->postscript();
  layout.footer = MetadataSection{
      layout.fileSize - layout.postscript.footerSize() - kPostscriptSize,
      layout.postscript.footerSize(),
      layout.postscript.footerCompressionType()};

  // Stripes metadata
  auto stripesMetadata = tablet->stripesMetadata();
  if (stripesMetadata.has_value()) {
    layout.stripes = *stripesMetadata;
  }

  // Stripe groups
  layout.stripeGroups = tablet->stripeGroupsMetadata();

  // Index partitions (from ClusterIndex if present).
  const auto* clusterIndex = tablet->clusterIndex();
  if (clusterIndex != nullptr) {
    const auto indexLayout = clusterIndex->layout(/*detail=*/false);
    layout.indexPartitions.reserve(indexLayout.partitions.size());
    for (const auto& part : indexLayout.partitions) {
      layout.indexPartitions.push_back(part.metadataSection);
    }
  }

  // Per-stripe info
  const auto stripeCount = tablet->stripeCount();
  layout.stripesInfo.reserve(stripeCount);
  std::vector<TabletReader::StreamLocation> locationsScratch;
  for (uint32_t i = 0; i < stripeCount; ++i) {
    auto stripeIdentifier = tablet->stripeIdentifier(i);
    locationsScratch.resize(tablet->streamCount(stripeIdentifier));
    tablet->streamLocations(stripeIdentifier, locationsScratch);
    auto stripeSize = std::accumulate(
        locationsScratch.begin(),
        locationsScratch.end(),
        0UL,
        [](auto size, const auto& location) { return size + location.size; });
    layout.stripesInfo.push_back({
        .offset = tablet->stripeOffset(i),
        .size = stripeSize,
        .stripeGroupIndex = stripeIdentifier.stripeGroup()->index(),
    });
  }

  // Optional sections
  layout.optionalSections = tablet->optionalSections();

  return layout;
}

FileLayout FileLayout::create(
    const std::string& path,
    velox::memory::MemoryPool* pool) {
  auto fs = velox::filesystems::getFileSystem(path, nullptr);
  auto file = fs->openFileForRead(path);
  return create(std::move(file), pool);
}

} // namespace facebook::nimble
