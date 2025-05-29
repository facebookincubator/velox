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

#include "velox/exec/FragmentResultCacheManager.h"

#include "common/file/FileInputStream.h"

namespace facebook::velox::exec {

void FragmentResultCacheManager::put(
    std::string planIdentifier,
    std::string splitIdentifier,
    std::vector<RowVectorPtr> result) {
  VELOX_CHECK(result.size() > 0, "Put empty result into FragmentResultCacheManager");

  /*
  IOBufOutputStream out(
      *pool_, nullptr, 64 * 1024);
  for (auto& rowVector : result) {
    auto rowType = asRowType(rowVector->type());
    auto numRows = rowVector->size();
    std::unique_ptr<velox::StreamArena> arena =
      std::make_unique<velox::StreamArena>(pool_.get());
    auto serializer =
      serde_->createIterativeSerializer(rowType, numRows, arena.get(), &serdeOptions_);
    serializer->append(rowVector);
    serializer->flush(&out);
  }
  auto iobuf = out.getIOBuf();
  auto path = "/path";
  auto fs = filesystems::getFileSystem(path, nullptr);
  auto file = fs->openFileForWrite(
      path,
      filesystems::FileOptions{
          {{}},
          nullptr,
          std::nullopt});
  file->append(std::move(iobuf));
  file->close();
  */
  if (cache_.count(CacheKey(planIdentifier, splitIdentifier)) > 0) {
    return;
  }
  cache_.emplace(CacheKey(planIdentifier, splitIdentifier), std::move(result));
  cache_list_.push_front(CacheKey(planIdentifier, splitIdentifier));
  if (cache_.size() == capacity_) {
    // Remove the least recently used element from the back of the list
    CacheKey lru_key = cache_list_.back();
    cache_.erase(lru_key);
    cache_list_.pop_back();
  }
}

bool FragmentResultCacheManager::get(
    std::string planIdentifier,
    std::string splitIdentifier,
    std::vector<RowVectorPtr>& result) {
  if (cache_.count(CacheKey(planIdentifier, splitIdentifier)) > 0) {
    result = cache_.at(CacheKey(planIdentifier, splitIdentifier));

    /*
    auto path = "/path";
    auto fs = filesystems::getFileSystem(path, nullptr);
    auto file = fs->openFileForRead(
        path,
        filesystems::FileOptions{
            {{}},
            nullptr,
            std::nullopt});
    std::unique_ptr<common::FileInputStream> input = std::make_unique<common::FileInputStream>(
        std::move(file), 64 * 1024, pool_.get());
    // TODO
    auto rowType = asRowType(result[0]->type());
    while (input->atEnd()) {
      // make empty row vector
      RowVectorPtr rowVector = std::make_shared<RowVector>(
        pool_.get(),
        rowType,
        nullptr, // nulls
        0,
        std::vector<VectorPtr>{});
      VectorStreamGroup::read(
        input.get(), pool_.get(), rowType, serde_.get(), &rowVector, {});
      result.push_back(rowVector);
    }
    */

    return true;
  }
  return false;
}

} // namespace facebook::velox::exec
