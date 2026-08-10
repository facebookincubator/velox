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
#pragma once

#include <cstdint>

namespace facebook::nimble {

// When encoding contains nested encodings, each nested encoding has its
// own identifier.
using NestedEncodingIdentifier = uint8_t;

struct EncodingIdentifiers {
  struct Dictionary {
    static constexpr NestedEncodingIdentifier Alphabet = 0;
    static constexpr NestedEncodingIdentifier Indices = 1;
  };

  struct MainlyConstant {
    static constexpr NestedEncodingIdentifier IsCommon = 0;
    static constexpr NestedEncodingIdentifier OtherValues = 1;
  };

  struct Nullable {
    static constexpr NestedEncodingIdentifier Data = 0;
    static constexpr NestedEncodingIdentifier Nulls = 1;
  };

  struct RunLength {
    static constexpr NestedEncodingIdentifier RunLengths = 0;
    static constexpr NestedEncodingIdentifier RunValues = 1;
  };

  struct SparseBool {
    static constexpr NestedEncodingIdentifier Indices = 0;
  };

  struct SharedDictionary {
    static constexpr NestedEncodingIdentifier Indices = 0;
  };

  struct Trivial {
    static constexpr NestedEncodingIdentifier Lengths = 0;
  };

  struct Delta {
    static constexpr NestedEncodingIdentifier Deltas = 0;
    static constexpr NestedEncodingIdentifier Restatements = 1;
    static constexpr NestedEncodingIdentifier IsRestatements = 2;
  };

  struct ALP {
    static constexpr NestedEncodingIdentifier EncodedValues = 0;
    static constexpr NestedEncodingIdentifier ExceptionPositions = 1;
    static constexpr NestedEncodingIdentifier ExceptionValues = 2;
  };

  struct FrequencyPartition {
    // Partition metadata
    // Eventually we may want to allow for non-power-of-two bit partitions, but
    // for now we can just define constants for the power-of-two cases.
    static constexpr NestedEncodingIdentifier PartitionOffsets = 0;
    static constexpr NestedEncodingIdentifier PartitionSizes = 1;
    // Per-tier dictionaries (1-bit, 2-bit, 4-bit, 8-bit, etc.)
    static constexpr NestedEncodingIdentifier Dict1Bit = 2;
    static constexpr NestedEncodingIdentifier Dict2Bit = 3;
    static constexpr NestedEncodingIdentifier Dict4Bit = 4;
    static constexpr NestedEncodingIdentifier Dict8Bit = 5;
    static constexpr NestedEncodingIdentifier Dict16Bit = 6;
    static constexpr NestedEncodingIdentifier Dict32Bit = 7;
    // Per-tier encoded keys
    static constexpr NestedEncodingIdentifier Keys1Bit = 8;
    static constexpr NestedEncodingIdentifier Keys2Bit = 9;
    static constexpr NestedEncodingIdentifier Keys4Bit = 10;
    static constexpr NestedEncodingIdentifier Keys8Bit = 11;
    static constexpr NestedEncodingIdentifier Keys16Bit = 12;
    static constexpr NestedEncodingIdentifier Keys32Bit = 13;
    // Unencoded partition (raw values that don't fit in any tier)
    static constexpr NestedEncodingIdentifier UnencodedValues = 14;
  };

  struct Pfor {
    // Nested sub-streams for PFOR: the ascending exception positions and the
    // exception residual values.
    static constexpr NestedEncodingIdentifier ExceptionPositions = 0;
    static constexpr NestedEncodingIdentifier ExceptionValues = 1;
  };

  struct BlockBitPacking {
    // Nested per-block metadata sub-streams for BlockBitPacking: the per-block
    // baselines, the per-block bit widths, and the ascending per-block data
    // offsets.
    static constexpr NestedEncodingIdentifier Baselines = 0;
    static constexpr NestedEncodingIdentifier BitWidths = 1;
    static constexpr NestedEncodingIdentifier Offsets = 2;
  };

  struct For {
    static constexpr NestedEncodingIdentifier BitWidths = 0;
    static constexpr NestedEncodingIdentifier References = 1;
    static constexpr NestedEncodingIdentifier BitOffsets = 2;
  };

  struct Fsst {
    static constexpr NestedEncodingIdentifier Lengths = 0;
  };
};

} // namespace facebook::nimble
