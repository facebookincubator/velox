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
#include "velox/dwio/nimble/tablet/Postscript.h"

#include <cstring>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/Constants.h"

namespace facebook::nimble {

Postscript::Postscript(
    uint32_t footerSize,
    CompressionType footerCompressionType,
    ChecksumType checksumType,
    uint32_t majorVersion,
    uint32_t minorVersion)
    : footerSize_(footerSize),
      footerCompressionType_(footerCompressionType),
      checksum_(0),
      checksumType_(checksumType),
      majorVersion_(majorVersion),
      minorVersion_(minorVersion) {}

Postscript Postscript::parse(std::string_view data) {
  NIMBLE_CHECK_GE(data.size(), kSize, "Invalid postscript length");

  Postscript ps;
  auto pos = data.data() + data.size() - 2;
  const uint16_t magicNumber = *reinterpret_cast<const uint16_t*>(pos);

  NIMBLE_CHECK_EQ(
      magicNumber, kMagicNumber, "Magic number mismatch. Not a nimble file!");

  pos -= 4;
  ps.majorVersion_ = *reinterpret_cast<const uint16_t*>(pos);
  ps.minorVersion_ = *reinterpret_cast<const uint16_t*>(pos + 2);

  NIMBLE_CHECK_LE(ps.majorVersion_, kVersionMajor, "Unsupported file version");

  pos -= 14;
  ps.footerSize_ = *reinterpret_cast<const uint32_t*>(pos);

  static_assert(sizeof(CompressionType) == 1);
  ps.footerCompressionType_ =
      *reinterpret_cast<const CompressionType*>(pos + 4);

  static_assert(sizeof(ChecksumType) == 1);
  ps.checksumType_ = *reinterpret_cast<const ChecksumType*>(pos + 5);
  ps.checksum_ = *reinterpret_cast<const uint64_t*>(pos + 6);
  return ps;
}

std::string Postscript::serialize() const {
  std::string buf(kSize, '\0');
  auto* pos = buf.data();
  std::memcpy(pos, &footerSize_, 4);
  pos += 4;
  std::memcpy(pos, &footerCompressionType_, 1);
  pos += 1;
  std::memcpy(pos, &checksumType_, 1);
  pos += 1;
  std::memcpy(pos, &checksum_, 8);
  pos += 8;
  std::memcpy(pos, &majorVersion_, 2);
  pos += 2;
  std::memcpy(pos, &minorVersion_, 2);
  pos += 2;
  std::memcpy(pos, &kMagicNumber, 2);
  return buf;
}

} // namespace facebook::nimble
