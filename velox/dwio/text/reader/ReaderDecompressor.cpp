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

#include <cstdint>

#include "velox/dwio/text/reader/ReaderDecompressor.h"

namespace velox::dwio::text::compression {

/// TODO: Add implementation
bool ReaderDecompressor::iBufRefill() {
  return false;
}

/// TODO: Add implementation
void ReaderDecompressor::initZlib() {
  return;
}

/// TODO: Add implementation
bool ReaderDecompressor::bufRefill() {
  return false;
}

/// TODO: Add implementation
void ReaderDecompressor::endZlib() {
  return;
}

/// TODO: Add implementation
void ReaderDecompressor::read(
    void* /*b*/,
    uint64_t /*length*/,
    uint64_t /*offset*/,
    facebook::velox::dwio::common::LogType /*type*/) {
  return;
}

/// TODO: Add implementation
ReaderDecompressor::ReaderDecompressor(
    // utils::PreloadableReader& s,
    CompressionKind /*k*/,
    MemoryPool& p,
    bool /*hiveDefaultMode*/)
    : pool(p) {
  return;
}

/// TODO: Add implementation
ReaderDecompressor::~ReaderDecompressor() {
  return;
}

/// TODO: Add implementation
uint64_t ReaderDecompressor::getLength() const {
  return 0;
}

/// TODO: Add implementation
uint64_t ReaderDecompressor::getNaturalReadSize() const {
  return 0;
}

} // namespace velox::dwio::text::compression
