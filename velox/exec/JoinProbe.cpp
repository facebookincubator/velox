
constexpr int32_t kMaxProbeBatch = 2048;
constexpr uint32_t kHitIdxMask = 0xe0000000U;
constexpr uint32_t kRowNumberMask = ~kHitIdxMask;
constexpr int32_t kFirstHit = 7 << 29;

constexpr uint8_t kNoMoreMatches = 255;

struct ProbeBatchState {
  bool useRowMask{false};
  // Number of build side rows to look at.
  int32_t numToCompare;

  /// Number of build side rows that missed the key compare.
  int32_t numToAdvance;

  /// Row numbers in probe key vectors for rows to compare.
  int32_t keyIdxToCompare[kMaxProbeBatch];

  /// Pointer to build side row to compare. 1:1 with 'keyIdxToCompare'.
  char* rowToCompare[kMaxProbeBatch];

  /// Probe key vector row number for rows where key compare missed. Need to
  /// look at next tag match in table if any.
  int32_t keyIdxToAdvance[kMaxProbeBatch];

  uint8_t* bits{nullptr};
};

#define KEY_COMPARE_DISPATCH(TEMPLATE_FUNC, typeKind, ...)               \
  [&]() {                                                                \
    switch (typeKind) {                                                  \
      case TypeKind::TINYINT: {                                          \
        return TEMPLATE_FUNC<TypeKind::TINYINT>(__VA_ARGS__);            \
      }                                                                  \
      case TypeKind::SMALLINT: {                                         \
        return TEMPLATE_FUNC<TypeKind::SMALLINT>(__VA_ARGS__);           \
      }                                                                  \
      case TypeKind::INTEGER: {                                          \
        return TEMPLATE_FUNC<TypeKind::INTEGER>(__VA_ARGS__);            \
      }                                                                  \
      case TypeKind::BIGINT: {                                           \
        return TEMPLATE_FUNC<TypeKind::BIGINT>(__VA_ARGS__);             \
      }                                                                  \
      case TypeKind::VARCHAR:                                            \
      case TypeKind::VARBINARY: {                                        \
        return TEMPLATE_FUNC<TypeKind::VARCHAR>(__VA_ARGS__);            \
      }                                                                  \
      default:                                                           \
        VELOX_UNREACHABLE(                                               \
            "Unsupported value ID type: ", mapTypeKindToName(typeKind)); \
    }                                                                    \
  }()

template <bool useRowMask>
FOLLY_ALWAYS_INLINE void recordSimdHits(
    ProbeBatchState& state,
    int32_t& numPassed,
    xsimd::batch<int32_t> orgIndices,
    uint8_t hits,
    int32_t i) {
  if (hits == 0xff) {
    if (numPassed == i) {
      numPassed += 8;
      return;
    }
    orgIndices.store_unaligned(&state.keyIdxToCompare[numPassed]);
    numPassed += 8;
    return;
  }
  simd::filter(orgIndices, hits)
      .store_unaligned(&state.keyIdxToCompare[numPassed]);
  numPassed += __builtin_popcount(hits);
  uint16_t misses = hits ^ 0xff;
  if (useRowMask) {
    // The rows where the high bits are clear have no further hits.
    misses &= simd::toBitMask((orgIndices & ~kRowNumberMask) != 0);
  }
  if (misses) {
    if (state.bits) {
      while (misses) {
	auto nth = bits::getAndClearLastSetBit(misses);
	if (state.bits[i + nth] != kNoMoreMatches) {
	  state.keyIdxToAdvance[state.numToAdvance++] =
	    state.keyIdxToCompare[i + nth];
	}
      }
    } else {
      simd::filter(orgIndices, misses)
	.store_unaligned(&state.keyIdxToAdvance[state.numToAdvance]);
      state.numToAdvance += __builtin_popcount(misses);
    }
  }
}

template <bool kProbeDict, bool useRowMask>
int32_t compareKeysSimd32(
    const int32_t* probeValues,
    const int32_t* probeDict,
    int32_t columnOffset,
    ProbeBatchState& state) {
  constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
  int32_t i = 0;
  int32_t numPassed = 0;
  for (; i + kBatch <= state.numToCompare; i += kBatch) {
    auto indices =
        xsimd::batch<int32_t>::load_unaligned(state.keyIdxToCompare + i);
    auto orgIndices = indices;
    if (useRowMask) {
      indices &= kRowNumberMask;
    }
    if (kProbeDict) {
      indices = simd::gather(probeDict, indices);
    }
    auto first4 = simd::getHalf<int64_t, false>(indices);
    auto probes1 = simd::gather<int64_t, int64_t, 4>(
        reinterpret_cast<const int64_t*>(probeValues),
        first4);
    auto second4 = simd::getHalf<int64_t, true>(indices);
    auto probes2 = simd::gather<int64_t, int64_t, 4>(
        reinterpret_cast<const int64_t*>(probeValues),
        second4);
    auto rows1 = simd::loadGatherIndices<int64_t, int64_t>(
        reinterpret_cast<int64_t*>(&state.rowToCompare[i]));
    auto rows2 = simd::loadGatherIndices<int64_t, int64_t>(
        reinterpret_cast<int64_t*>(&state.rowToCompare[i + 4]));
    auto keys1 = simd::gather<int64_t, int64_t, 1>(
        reinterpret_cast<int64_t*>(columnOffset), rows1);
    auto keys2 = simd::gather<int64_t, int64_t, 1>(
        reinterpret_cast<int64_t*>(columnOffset), rows2);
    auto match1 =
        simd::toBitMask((probes1 & 0xffffffffL) == (keys1 & 0xffffffffL));
    auto match2 =
        simd::toBitMask((probes2 & 0xffffffffL) == (keys2 & 0xffffffffL));
    auto hits = match1 | (match2 << 4);
    recordSimdHits<useRowMask>(state, numPassed, orgIndices, hits, i);
  }
  state.numToCompare = numPassed;
  return i;
}

template <bool kProbeDict, bool useRowMask>
int32_t compareKeysSimd64(
    const int64_t* probeValues,
    const int32_t* probeDict,
    int32_t columnOffset,
    ProbeBatchState& state) {
  constexpr int32_t kBatch = xsimd::batch<int32_t>::size;

  int32_t i = 0;
  int32_t numPassed = 0;
  for (; i + kBatch <= state.numToCompare; i += kBatch) {
    auto indices =
        xsimd::batch<int32_t>::load_unaligned(state.keyIdxToCompare + i);
    auto orgIndices = indices;
    if (useRowMask) {
      indices &= kRowNumberMask;
    }
    if (kProbeDict) {
      indices = simd::gather(probeDict, indices);
    }
    auto first4 = simd::getHalf<int64_t, false>(indices);
    auto probes1 = simd::gather<int64_t, int64_t, 8>(
        reinterpret_cast<const int64_t*>(probeValues),
        first4);
    auto second4 = simd::getHalf<int64_t, true>(indices);
    auto probes2 = simd::gather<int64_t, int64_t, 8>(
        reinterpret_cast<const int64_t*>(probeValues),
        second4);
    auto rows1 = simd::loadGatherIndices<int64_t, int64_t>(
        reinterpret_cast<int64_t*>(&state.rowToCompare[i]));
    auto rows2 = simd::loadGatherIndices<int64_t, int64_t>(
        reinterpret_cast<int64_t*>(&state.rowToCompare[i + 4]));
    auto keys1 = simd::gather<int64_t, int64_t, 1>(
        reinterpret_cast<int64_t*>(columnOffset), rows1);
    auto keys2 = simd::gather<int64_t, int64_t, 1>(
        reinterpret_cast<int64_t*>(columnOffset), rows2);
    auto match1 = simd::toBitMask((probes1) == (keys1));
    auto match2 = simd::toBitMask((probes2) == (keys2));
    auto hits = match1 | (match2 << 4);
    recordSimdHits<useRowMask>(state, numPassed, orgIndices, hits, i);
  }
  state.numToCompare = numPassed;
  return i;
}

template <typename T, bool kProbeDict, bool useRowMask>
int32_t compareKeysSimd(
    const T* probeValues,
    const int32_t* probeDict,
    int32_t columnOffset,
    ProbeBatchState& state) {
  if (std::is_same_v<T, int64_t>) {
    return compareKeysSimd64<kProbeDict, useRowMask>(
        reinterpret_cast<const int64_t*>(probeValues),
        probeDict,
        columnOffset,
        state);
  }
  if (std::is_same_v<T, int32_t>) {
    return compareKeysSimd32<kProbeDict, useRowMask>(
        reinterpret_cast<const int32_t*>(probeValues),
        probeDict,
        columnOffset,
        state);
  }
  return 0;
}

template <typename T, bool kProbeDict, bool useRowMask>
void compareKeys(
    const T* probeValues,
    const int32_t* probeDict,
    int32_t columnOffset,
    ProbeBatchState& state) {
  auto numCompares = state.numToCompare;
  int32_t numPassed = 0;
  int32_t first = 0;
  if (FLAGS_simd_compare) {
    first = compareKeysSimd<T, kProbeDict, useRowMask>(
        probeValues, probeDict, columnOffset, state);
    numPassed = state.numToCompare;
  }
  for (auto i = first; i < numCompares; ++i) {
    char* buildRow = state.rowToCompare[i];
    auto buildValue = *reinterpret_cast<const T*>(buildRow + columnOffset);
    auto keyIdx = state.keyIdxToCompare[i];
    auto orgKeyIdx = keyIdx;
    if (useRowMask) {
      keyIdx &= kRowNumberMask;
    }
    int32_t probeRow = keyIdx;
    auto probeValue = reinterpret_cast<const T*>(
        probeValues)[kProbeDict ? probeDict[probeRow] : probeRow];
    if (probeValue == buildValue) {
      if (i != numPassed) {
        state.rowToCompare[numPassed] = buildRow;
        state.keyIdxToCompare[numPassed] = probeRow;
      }
      ++numPassed;
    } else {
      if (useRowMask) {
        if (orgKeyIdx & kHitIdxMask) {
          state.keyIdxToAdvance[state.numToAdvance++] = orgKeyIdx;
        }
      } else {
        if (state.bits) {
          if (state.bits[probeRow] != kNoMoreMatches) {
            state.keyIdxToAdvance[state.numToAdvance++] = probeRow;
          }
        } else {
          state.keyIdxToAdvance[state.numToAdvance++] = keyIdx;
        }
      }
    }
  }
  state.numToCompare = numPassed;
}

template <TypeKind Kind>
void compareFlatKeys(
    const BaseVector* base,
    int32_t columnOffset,
    ProbeBatchState& state) {
  using T = typename TypeTraits<Kind>::NativeType;
  const T* values = base->as<FlatVector<T>>()->rawValues();
  if (state.useRowMask) {
    compareKeys<T, false, true>(values, nullptr, columnOffset, state);
  } else {
    compareKeys<T, false, false>(values, nullptr, columnOffset, state);
  }
}

template <TypeKind Kind>
void compareDictKeys(
    const BaseVector* base,
    const int32_t* probeDict,
    int32_t columnOffset,
    ProbeBatchState& state) {
  using T = typename TypeTraits<Kind>::NativeType;
  const T* values;
  T value;
  if (base->encoding() == VectorEncoding::Simple::FLAT) {
    values = base->as<FlatVector<T>>()->rawValues();
  } else if (base->encoding() == VectorEncoding::Simple::CONSTANT) {
    value = base->as<ConstantVector<T>>()->valueAt(0);
    values = &value;
  }
  if (state.useRowMask) {
    compareKeys<T, true, true>(values, probeDict, columnOffset, state);
  } else {
    compareKeys<T, true, false>(values, probeDict, columnOffset, state);
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::preprobeTags(
    HashLookup& lookup,
    int32_t begin,
    int32_t end) {
  for (auto i = begin; i < end; ++i) {
    auto row = lookup.rows[i];
    auto hash = lookup.hashes[row];
    int64_t offset = bucketOffset(hash);
    __builtin_prefetch(bucketAt(offset));
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::preprobeSingles(
    HashLookup& lookup,
    int32_t begin,
    int32_t end) {
  for (auto i = begin; i < end; ++i) {
    auto row = lookup.rows[i];
    auto hash = lookup.hashes[row];
    __builtin_prefetch(&itemAt(hash &singleSizeMask_));
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::incrementHashTags(uint64_t& h) {
  h = (h & ~sizeMask_) | ((h + sizeof(Bucket)) & sizeMask_);
}

template <bool ignoreNullKeys>
uint64_t HashTable<ignoreNullKeys>::incrementHashSingle(
    uint64_t& h,
    int32_t n) {
  return h = (h & ~singleSizeMask_) |
      ((h + sizeof(int64_t) * n) & singleSizeMask_);
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::compareAllKeys(
    HashLookup& lookup,
    ProbeBatchState& state) {
  state.numToAdvance = 0;
  for (auto i = 0; i < lookup.hashers.size(); ++i) {
    if (state.numToCompare == 0) {
      return;
    }
    auto& decoded = lookup.hashers[i]->decodedVector();
    const BaseVector* base = decoded.base();
    auto columnOffset = rows_->columnAt(i).offset();
    if (decoded.isIdentityMapping()) {
      KEY_COMPARE_DISPATCH(
          compareFlatKeys, base->typeKind(), base, columnOffset, state);
    } else {
      KEY_COMPARE_DISPATCH(
          compareDictKeys,
          base->typeKind(),
          base,
          decoded.indices(),
          columnOffset,
          state);
    }
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::probeBatchTags(
    HashLookup& lookup,
    int32_t begin,
    int32_t end) {
  ProbeBatchState state;
  state.numToCompare = 0;
  state.numToAdvance = 0;

  lookup.nthBit.resize(lookup.hashes.size());
  auto* keyIndices = lookup.rows.data() + begin;
  bool prefetchRow = (FLAGS_hash_prefetch_mode & 2) != 0;
  if (FLAGS_hash_prefetch_mode & 1) {
    preprobeTags(lookup, begin, end);
  }
  auto numProbe = end - begin;
  constexpr uint8_t kEmptyTag = 0;
  const auto kEmptyGroup = BaseHashTable::TagVector::broadcast(kEmptyTag);
  state.bits = lookup.nthBit.data();
  // Loop until all probes have hit or missed.
  bool isFirst = true;
  for (;;) {
    for (auto i = 0; i < numProbe; ++i) {
      auto row = keyIndices[i];
      auto hash = lookup.hashes[row];
      int64_t bucket = bucketOffset(hash);
      auto tags =
          BaseHashTable::loadTags(reinterpret_cast<uint8_t*>(table_), bucket);
      const auto tag = BaseHashTable::hashTag(hash);
      auto wantedTags = BaseHashTable::TagVector::broadcast(tag);
      for (;;) {
        uint16_t bits = simd::toBitMask(tags == wantedTags);
        if (!isFirst) {
          bits &= ~bits::lowMask(state.bits[row]);
        }
        if (bits) {
          auto pos = bits::getAndClearLastSetBit(bits);
          state.keyIdxToCompare[state.numToCompare] = row;
          state.rowToCompare[state.numToCompare] =
              bucketAt(bucket)->pointerAt(pos);
          if (prefetchRow) {
            __builtin_prefetch(state.rowToCompare[state.numToCompare]);
          }
          uint8_t nextBits = kNoMoreMatches;
          if (bits) {
            // There are more in the bucket. Next one will be above current.
            nextBits = pos + 1;
          } else if (0 == simd::toBitMask(tags == kEmptyGroup)) {
            // There are no more hits and no gaps. Next time look at next
            // bucket.
            nextBits = 0;
            incrementHashTags(lookup.hashes[row]);
          }
          state.bits[row] = nextBits;
          ++state.numToCompare;
          break;
        } else {
          // No hit in this bucket.
          if (simd::toBitMask(tags == kEmptyGroup)) {
            break;
          }
          incrementHashTags(lookup.hashes[row]);
          bucket = nextBucketOffset(bucket);
          tags = BaseHashTable::loadTags(
              reinterpret_cast<uint8_t*>(table_), bucket);
        }
      }
    }
    compareAllKeys(lookup, state);
    if (lookup.makeDenseHits) {
      for (auto i = 0; i < state.numToCompare; ++i) {
        lookup.hits[lookup.hitRows.size()] = state.rowToCompare[i];
        lookup.hitRows.push_back(state.keyIdxToCompare[i]);
      }
    } else {
      for (auto i = 0; i < state.numToCompare; ++i) {
        lookup.hits[state.keyIdxToCompare[i]] = state.rowToCompare[i];
      }
    }
    if (state.numToAdvance == 0) {
      break;
    }
    // Look up next hit for the tag matches that did not match keys.
    state.numToCompare = 0;
    keyIndices = state.keyIdxToAdvance;
    numProbe = state.numToAdvance;

    isFirst = false;
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::probeBatchSingles(
    HashLookup& lookup,
    int32_t begin,
    int32_t end) {
  ProbeBatchState state;
  state.numToCompare = 0;
  state.numToAdvance = 0;

  auto* keyIndices = lookup.rows.data() + begin;
  bool prefetchRow = (FLAGS_hash_prefetch_mode & 2) != 0;
  if (FLAGS_hash_prefetch_mode & 1) {
    preprobeSingles(lookup, begin, end);
  }
  auto numProbe = end - begin;
  // Loop until all probes have hit or missed.
  for (;;) {
    for (auto i = 0; i < numProbe; ++i) {
      auto row = keyIndices[i];
      auto hash = lookup.hashes[row];
      auto offset = hash & singleSizeMask_;
      for (;;) {
      auto item = itemAt(offset);
        if (!item) {
          break;
        }
        if (tagMatch(item, hash)) {
          state.keyIdxToCompare[state.numToCompare] = row;
          state.rowToCompare[state.numToCompare] =
              reinterpret_cast<char*>(item & kPointerMask);
          if (prefetchRow) {
            __builtin_prefetch(state.rowToCompare[state.numToCompare]);
          }
	  ++state.numToCompare;
	  break;
        }
	offset = incrementHashSingle(lookup.hashes[row]) & singleSizeMask_;
      }
    }
    compareAllKeys(lookup, state);
    if (lookup.makeDenseHits) {
      for (auto i = 0; i < state.numToCompare; ++i) {
        lookup.hits[lookup.hitRows.size()] = state.rowToCompare[i];
        lookup.hitRows.push_back(state.keyIdxToCompare[i]);
      }
    } else {
      for (auto i = 0; i < state.numToCompare; ++i) {
        lookup.hits[state.keyIdxToCompare[i]] = state.rowToCompare[i];
      }
    }
    if (state.numToAdvance == 0) {
      break;
    }
    // Look up next hit for the tag matches that did not match keys.
    state.numToCompare = 0;
    keyIndices = state.keyIdxToAdvance;
    numProbe = state.numToAdvance;
    for (auto i = 0; i < numProbe; ++i) {
      incrementHashSingle(lookup.hashes[keyIndices[i]]);
    }
  }
}

int64_t lookFor = 1959;

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::probeBatchSinglesSimd(
    HashLookup& lookup,
    int32_t begin,
    int32_t end) {
  ProbeBatchState state;
#if 0
  auto expect = folly::hasher<int64_t>()(lookFor);
  for (auto i = begin; i < end; ++i) {
    if (lookup.hashes[i] == expect) {
      printf("***bing %d\n", i - begin);
    }
  }
#endif
  state.useRowMask = true;
  state.numToCompare = 0;
  state.numToAdvance = 0;
  bool isFirst = true;
  auto* keyIndices = lookup.rows.data() + begin;
  bool prefetchRow = (FLAGS_hash_prefetch_mode & 2) != 0;
  if (FLAGS_hash_prefetch_mode & 1) {
    preprobeSingles(lookup, begin, end);
  }
  auto numProbe = end - begin;
  static int32_t iotaInts[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  auto iota = xsimd::batch<int32_t>::load_unaligned(&iotaInts[0]);
  // Loop until all probes have hit or missed.
  for (;;) {
    constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
    xsimd::batch_bool<int64_t> mask1 = true;
    xsimd::batch_bool<int64_t> mask2 = true;
    bool isDense = lookup.rows[end - 1] - lookup.rows[begin] == end - begin - 1;
    for (auto i = 0; i < numProbe; i += kBatch) {
      uint16_t undecided;
      if (isFirst) {
        xsimd::batch<int32_t> indices;
        xsimd::batch<int64_t> hashes1;
        xsimd::batch<int64_t> hashes2;
        bool isPartial = i + 8 > numProbe;
        if (isPartial) {
          mask1 =
              simd::leadingMask<int64_t>(i + 4 <= numProbe ? 4 : numProbe - i);
          mask2 = simd::leadingMask<int64_t>(
              i + 8 <= numProbe       ? 0
                  : i + 4 >= numProbe ? 0
                                      : numProbe - (i + 4));
        }
        if (isDense & !isPartial) {
          hashes1 =
              xsimd::batch<int64_t>::load_unaligned(&lookup.hashes[i + begin]);
          hashes2 = xsimd::batch<int64_t>::load_unaligned(
              &lookup.hashes[i + 4 + begin]);
          indices = iota + (i + begin);
        } else {
          indices = xsimd::batch<int32_t>::load_unaligned(keyIndices + i);
          auto first4 =
              simd::loadGatherIndices<int64_t, int32_t>(keyIndices + i);
          hashes1 = simd::maskGather<int64_t, int32_t>(
              xsimd::broadcast<int64_t>(0),
              mask1,
              reinterpret_cast<int64_t*>(lookup.hashes.data()),
              first4);
          auto second4 =
              simd::loadGatherIndices<int64_t, int32_t>(keyIndices + i + 4);
          hashes2 = simd::maskGather<int64_t, int32_t>(
              xsimd::broadcast<int64_t>(0),
              mask2,
              reinterpret_cast<int64_t*>(lookup.hashes.data()),
              second4);
        }
        auto offsets1 = hashes1 & static_cast<int64_t>(singleSizeMask_);
        auto items1 = simd::maskGather<int64_t, int64_t, 1>(
            xsimd::broadcast<int64_t>(0),
            mask1,
            reinterpret_cast<int64_t*>(table_),
            offsets1);
        auto offsets2 = hashes2 & static_cast<int64_t>(singleSizeMask_);
        auto items2 = simd::maskGather<int64_t, int64_t, 1>(
            xsimd::broadcast<int64_t>(0),
            mask2,
            reinterpret_cast<int64_t*>(table_),
            offsets2);

        auto tagMatch1 =
            simd::toBitMask(((items1 ^ hashes1) & ~kPointerMask) == 0L);
        auto empty1 = simd::toBitMask(items1 == 0L);
        tagMatch1 &= ~empty1;
        auto tagMatch2 =
            simd::toBitMask(((items2 ^ hashes2) & ~kPointerMask) == 0L);
        auto empty2 = simd::toBitMask(items2 == 0L);
        tagMatch2 &= ~empty2;
        uint16_t empties = empty1 | (empty2 << 4);
        uint16_t hits = tagMatch1 | (tagMatch2 << 4);
        if (tagMatch1) {
          simd::filter(items1 & kPointerMask, tagMatch1)
              .store_unaligned(reinterpret_cast<int64_t*>(
                  &state.rowToCompare[state.numToCompare]));
        }
        if (tagMatch2) {
          simd::filter(items2 & kPointerMask, tagMatch2)
              .store_unaligned(reinterpret_cast<int64_t*>(
                  &state.rowToCompare
                       [state.numToCompare + __builtin_popcount(tagMatch1)]));
        }
        // Mark the first hits.
        if (hits) {
          (simd::filter(indices, hits) | kFirstHit)
              .store_unaligned(&state.keyIdxToCompare[state.numToCompare]);
          int32_t numHits = __builtin_popcount(hits);
          if (prefetchRow) {
            for (auto i = 0; i < numHits; ++i) {
              __builtin_prefetch(state.rowToCompare[state.numToCompare + i]);
            }
          }
          state.numToCompare += numHits;
        }
        // Not a tag match and not an empty.
        undecided = (~hits & ~empties) & 0xff;
      } else {
        undecided = numProbe + kBatch <= i ? 0xff : bits::lowMask(numProbe - i);
      }
      while (undecided) {
        int32_t nth = bits::getAndClearLastSetBit(undecided);
        int32_t row = keyIndices[i + nth];
        uint32_t hitIdx = 0;
        uint64_t hash = lookup.hashes[row & kRowNumberMask];
        if (isFirst) {
          // If the initial compare was undecided, check the containing group
          // of 4.
          hash &= ~31UL;
          lookup.hashes[row] = hash;
        } else {
          hitIdx = row & kHitIdxMask;
          row &= kRowNumberMask;
          if (hitIdx == kFirstHit) {
            // Tag hit from first gather missed. Reload the containing group
            // of 4.
            hash &= ~31UL;
            lookup.hashes[row] = hash;
            hitIdx = 0;
          } else {
            // This is used to mask out hits below hitIdx. So, if the
            // hit was lane 0, hitIdx is 1 so that ~lowMask is ~1 to
            // mask out low bit.
            hitIdx = (hitIdx >> 29);
          }
        }
        auto tagWords = xsimd::broadcast<int64_t>(hash);
        for (;;) {
          int64_t offset = hash & singleSizeMask_;
          auto items = xsimd::batch<int64_t>::load_unaligned(
              reinterpret_cast<int64_t*>(&itemAt(offset)));
          uint16_t tags =
              simd::toBitMask(((tagWords ^ items) & ~kPointerMask) == 0);
          uint16_t empties = simd::toBitMask(items == 0L);
          if (empties && (tags &= ~empties) == 0) {
            break;
          }
          tags &= ~bits::lowMask(hitIdx);
          if (tags) {
            uint32_t lane = bits::getAndClearLastSetBit(tags);
            uint32_t hitInfo = (tags == 0 && empties) ? 0 : ((lane + 1) << 29);
            if (hitInfo && lane == 3) {
              hitInfo = kFirstHit;
              incrementHashSingle(lookup.hashes[row], 4);
            }
            state.keyIdxToCompare[state.numToCompare] = row | hitInfo;
            auto buildRow = reinterpret_cast<char*>(
                itemAt(offset + lane * sizeof(int64_t)) & kPointerMask);
            if (prefetchRow) {
              __builtin_prefetch(buildRow);
            }
            state.rowToCompare[state.numToCompare] = buildRow;
            ++state.numToCompare;
            break;
          }
          if (empties) {
            break;
          }
          hitIdx = 0;
          hash = incrementHashSingle(lookup.hashes[row], 4);
        }
      }
    }
    compareAllKeys(lookup, state);
    if (lookup.makeDenseHits) {
      for (auto i = 0; i < state.numToCompare; ++i) {
        lookup.hits[lookup.hitRows.size()] = state.rowToCompare[i];
        lookup.hitRows.push_back(state.keyIdxToCompare[i] & kRowNumberMask);
      }
    } else {
      for (auto i = 0; i < state.numToCompare; ++i) {
        lookup.hits[state.keyIdxToCompare[i] & kRowNumberMask] =
            state.rowToCompare[i];
      }
    }
    if (state.numToAdvance == 0) {
      break;
    }
    // Look up next hit for the tag matches that did not match keys.
    state.numToCompare = 0;
    keyIndices = state.keyIdxToAdvance;
    numProbe = state.numToAdvance;
    isFirst = false;
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::joinProbe2(HashLookup& lookup) {
  if (!lookup.makeDenseHits) {
    std::fill(lookup.hits.begin(), lookup.hits.end(), nullptr);
  }
  int32_t batch = FLAGS_probe_batch;
  auto numProbe = lookup.rows.size();
  for (auto begin = 0; begin < numProbe; begin += batch) {
    switch (FLAGS_join_mode) {
      case 1:
        probeBatchTags(
            lookup, begin, std::min<int32_t>(begin + batch, numProbe));
        break;
      case 2:
        probeBatchSingles(
            lookup, begin, std::min<int32_t>(begin + batch, numProbe));
        break;
      case 3:
        probeBatchSinglesSimd(
            lookup, begin, std::min<int32_t>(begin + batch, numProbe));
        break;
      default:
        VELOX_UNREACHABLE();
    }
  }
}

auto ldm(int32_t n) {
  return simd::leadingMask<int64_t>(n);
}
