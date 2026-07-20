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

#include "velox/connectors/hive/ExtractionUtils.h"

#include "velox/type/Filter.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::hive {

namespace {

/// Extract sizes from a MapVector or ArrayVector as a FlatVector<int64_t>.
VectorPtr extractSizes(const VectorPtr& input, memory::MemoryPool* pool) {
  auto numRows = input->size();
  auto sizesBuf = AlignedBuffer::allocate<int64_t>(numRows, pool);
  auto* sizesData = sizesBuf->asMutable<int64_t>();

  const vector_size_t* rawSizes;
  if (input->typeKind() == TypeKind::MAP) {
    rawSizes = input->as<MapVector>()->rawSizes();
  } else {
    rawSizes = input->as<ArrayVector>()->rawSizes();
  }
  for (vector_size_t i = 0; i < numRows; ++i) {
    sizesData[i] = rawSizes[i];
  }

  return std::make_shared<FlatVector<int64_t>>(
      pool,
      BIGINT(),
      input->nulls(),
      numRows,
      std::move(sizesBuf),
      std::vector<BufferPtr>{});
}

/// Filter a MapVector to only keep entries with matching keys.
VectorPtr filterMapByKeys(
    const VectorPtr& input,
    const ExtractionPathElement& step,
    memory::MemoryPool* pool) {
  auto* map = input->as<MapVector>();
  auto numRows = map->size();
  auto* rawOffsets = map->rawOffsets();
  auto* rawSizes = map->rawSizes();
  auto mapKeys = map->mapKeys();
  auto mapValues = map->mapValues();

  // Build output offsets and sizes, and a mapping of which key-value pairs
  // to keep.
  auto newOffsetsBuf = AlignedBuffer::allocate<vector_size_t>(numRows, pool);
  auto newSizesBuf = AlignedBuffer::allocate<vector_size_t>(numRows, pool);
  auto* newOffsets = newOffsetsBuf->asMutable<vector_size_t>();
  auto* newSizes = newSizesBuf->asMutable<vector_size_t>();

  // Build keep flags for all key-value pairs across all map entries.
  std::vector<bool> keepFlags;
  vector_size_t totalInputElements = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    totalInputElements += rawSizes[i];
  }
  keepFlags.resize(totalInputElements, false);

  if (totalInputElements == 0) {
    // All maps are empty; return the input as-is with zeroed sizes.
    return std::make_shared<MapVector>(
        pool,
        input->type(),
        map->nulls(),
        numRows,
        std::move(newOffsetsBuf),
        std::move(newSizesBuf),
        mapKeys,
        mapValues);
  }

  auto* stringFilter =
      dynamic_cast<const StringMapKeyFilterExtractionPathElement*>(&step);
  if (stringFilter) {
    std::unordered_set<std::string> keySet(
        stringFilter->filterKeys().begin(), stringFilter->filterKeys().end());
    auto* keyVector = mapKeys->as<SimpleVector<StringView>>();
    for (vector_size_t i = 0; i < totalInputElements; ++i) {
      VELOX_DCHECK_LT(
          static_cast<size_t>(i), keepFlags.size(), "Index out of bounds");
      if (!keyVector->isNullAt(i) &&
          keySet.count(std::string(keyVector->valueAt(i))) > 0) {
        keepFlags[i] = true;
      }
    }
  } else {
    auto& intFilter =
        static_cast<const IntMapKeyFilterExtractionPathElement&>(step);
    std::unordered_set<int64_t> keySet(
        intFilter.filterKeys().begin(), intFilter.filterKeys().end());
    auto* keyVector = mapKeys->as<SimpleVector<int64_t>>();
    for (vector_size_t i = 0; i < totalInputElements; ++i) {
      VELOX_DCHECK_LT(
          static_cast<size_t>(i), keepFlags.size(), "Index out of bounds");
      if (!keyVector->isNullAt(i) && keySet.count(keyVector->valueAt(i)) > 0) {
        keepFlags[i] = true;
      }
    }
  }

  // Count kept entries per map.
  vector_size_t totalKept = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    newOffsets[i] = totalKept;
    vector_size_t kept = 0;
    for (vector_size_t j = 0; j < rawSizes[i]; ++j) {
      VELOX_DCHECK_LT(
          static_cast<size_t>(rawOffsets[i] + j),
          keepFlags.size(),
          "Index out of bounds");
      if (keepFlags[rawOffsets[i] + j]) {
        ++kept;
      }
    }
    newSizes[i] = kept;
    totalKept += kept;
  }

  // Build index mapping for kept entries.
  auto indexBuf = AlignedBuffer::allocate<vector_size_t>(totalKept, pool);
  auto* indices = indexBuf->asMutable<vector_size_t>();
  vector_size_t outIdx = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    for (vector_size_t j = 0; j < rawSizes[i]; ++j) {
      auto srcIdx = rawOffsets[i] + j;
      VELOX_DCHECK_LT(
          static_cast<size_t>(srcIdx), keepFlags.size(), "Index out of bounds");
      if (keepFlags[srcIdx]) {
        indices[outIdx++] = srcIdx;
      }
    }
  }

  // Create filtered keys and values using dictionary wrapping.
  auto filteredKeys =
      BaseVector::wrapInDictionary(nullptr, indexBuf, totalKept, mapKeys);
  auto filteredValues =
      BaseVector::wrapInDictionary(nullptr, indexBuf, totalKept, mapValues);

  return std::make_shared<MapVector>(
      pool,
      input->type(),
      map->nulls(),
      numRows,
      std::move(newOffsetsBuf),
      std::move(newSizesBuf),
      filteredKeys,
      filteredValues);
}

/// Recursive implementation of extraction chain application.
VectorPtr applyExtractionChainImpl(
    const VectorPtr& input,
    const std::vector<ExtractionPathElementPtr>& chain,
    size_t index,
    memory::MemoryPool* pool) {
  if (index >= chain.size()) {
    return input;
  }

  const auto& step = chain[index];
  switch (step->step()) {
    case ExtractionStep::kStructField: {
      auto* row = input->as<RowVector>();
      VELOX_CHECK_NOT_NULL(row);
      auto& fieldName =
          static_cast<const StructFieldExtractionPathElement&>(*step)
              .fieldName();
      auto childIdx = input->type()->asRow().getChildIdx(fieldName);
      return applyExtractionChainImpl(
          row->childAt(childIdx), chain, index + 1, pool);
    }

    case ExtractionStep::kMapKeys: {
      auto* map = input->as<MapVector>();
      VELOX_CHECK_NOT_NULL(map);
      if (index + 1 >= chain.size()) {
        return std::make_shared<ArrayVector>(
            pool,
            ARRAY(map->mapKeys()->type()),
            map->nulls(),
            map->size(),
            map->offsets(),
            map->sizes(),
            map->mapKeys());
      }
      VELOX_CHECK_EQ(
          static_cast<int>(chain[index + 1]->step()),
          static_cast<int>(ExtractionStep::kArrayElements));
      auto transformedKeys =
          applyExtractionChainImpl(map->mapKeys(), chain, index + 2, pool);
      return std::make_shared<ArrayVector>(
          pool,
          ARRAY(transformedKeys->type()),
          map->nulls(),
          map->size(),
          map->offsets(),
          map->sizes(),
          transformedKeys);
    }

    case ExtractionStep::kMapValues: {
      auto* map = input->as<MapVector>();
      VELOX_CHECK_NOT_NULL(map);
      if (index + 1 >= chain.size()) {
        return std::make_shared<ArrayVector>(
            pool,
            ARRAY(map->mapValues()->type()),
            map->nulls(),
            map->size(),
            map->offsets(),
            map->sizes(),
            map->mapValues());
      }
      VELOX_CHECK_EQ(
          static_cast<int>(chain[index + 1]->step()),
          static_cast<int>(ExtractionStep::kArrayElements));
      auto transformedValues =
          applyExtractionChainImpl(map->mapValues(), chain, index + 2, pool);
      return std::make_shared<ArrayVector>(
          pool,
          ARRAY(transformedValues->type()),
          map->nulls(),
          map->size(),
          map->offsets(),
          map->sizes(),
          transformedValues);
    }

    case ExtractionStep::kMapKeyFilter: {
      auto filtered = filterMapByKeys(input, *step, pool);
      return applyExtractionChainImpl(filtered, chain, index + 1, pool);
    }

    case ExtractionStep::kArrayElements: {
      auto* array = input->as<ArrayVector>();
      VELOX_CHECK_NOT_NULL(array);
      if (index + 1 >= chain.size()) {
        return input;
      }
      auto transformedElements =
          applyExtractionChainImpl(array->elements(), chain, index + 1, pool);
      return std::make_shared<ArrayVector>(
          pool,
          ARRAY(transformedElements->type()),
          array->nulls(),
          array->size(),
          array->offsets(),
          array->sizes(),
          transformedElements);
    }

    case ExtractionStep::kSize: {
      return extractSizes(input, pool);
    }
  }
  VELOX_UNREACHABLE();
}

/// Analyze extraction chains on a ROW type to determine which fields are
/// needed.
std::unordered_set<std::string> analyzeStructNeeds(
    const std::vector<NamedExtraction>& extractions) {
  std::unordered_set<std::string> neededFields;
  for (const auto& extraction : extractions) {
    if (extraction.chain.empty()) {
      // Pass-through: need all fields.
      return {};
    }
    if (extraction.chain[0]->step() == ExtractionStep::kStructField) {
      neededFields.insert(
          static_cast<const StructFieldExtractionPathElement&>(
              *extraction.chain[0])
              .fieldName());
    } else {
      // Non-struct step on a struct: need everything.
      return {};
    }
  }
  return neededFields;
}

/// Build sub-chains by stripping the first step from each extraction chain.
/// For MapKeys/MapValues, also strips the following ArrayElements step.
/// Only includes extractions whose first step matches 'firstStep'.
/// For ROW StructField, only includes extractions targeting 'fieldName'.
std::vector<NamedExtraction> buildSubChains(
    const std::vector<NamedExtraction>& extractions,
    ExtractionStep firstStep,
    const std::string& fieldName = "") {
  std::vector<NamedExtraction> subChains;
  for (const auto& extraction : extractions) {
    if (extraction.chain.empty()) {
      continue;
    }
    if (extraction.chain[0]->step() != firstStep) {
      continue;
    }
    if (firstStep == ExtractionStep::kStructField) {
      const auto& name = static_cast<const StructFieldExtractionPathElement&>(
                             *extraction.chain[0])
                             .fieldName();
      if (name != fieldName) {
        continue;
      }
    }

    // Determine how many leading steps to skip.
    size_t skip = 1;
    if ((firstStep == ExtractionStep::kMapKeys ||
         firstStep == ExtractionStep::kMapValues) &&
        skip < extraction.chain.size() &&
        extraction.chain[skip]->step() == ExtractionStep::kArrayElements) {
      ++skip;
    }

    if (skip < extraction.chain.size()) {
      NamedExtraction sub;
      sub.outputName = extraction.outputName;
      sub.chain = std::vector<ExtractionPathElementPtr>(
          extraction.chain.begin() + static_cast<ptrdiff_t>(skip),
          extraction.chain.end());
      sub.dataType = extraction.dataType;
      subChains.push_back(std::move(sub));
    }
  }
  return subChains;
}

} // namespace

VectorPtr applyExtractionChain(
    const VectorPtr& input,
    const std::vector<ExtractionPathElementPtr>& chain,
    memory::MemoryPool* pool) {
  return applyExtractionChainImpl(input, chain, 0, pool);
}

void configureExtractionScanSpec(
    const TypePtr& hiveType,
    const std::vector<NamedExtraction>& extractions,
    common::ScanSpec& spec,
    memory::MemoryPool* pool) {
  if (extractions.empty()) {
    return;
  }

  switch (hiveType->kind()) { // NOLINT(clang-diagnostic-switch-enum)
    case TypeKind::MAP: {
      // Determine the extraction type from the first step of all chains.
      bool allKeys = true;
      bool allValues = true;
      bool allSize = true;
      bool allMapKeyFilter = true;
      for (const auto& extraction : extractions) {
        if (extraction.chain.empty()) {
          allKeys = false;
          allValues = false;
          allSize = false;
          allMapKeyFilter = false;
          break;
        }
        auto firstStep = extraction.chain[0]->step();
        if (firstStep != ExtractionStep::kMapKeys) {
          allKeys = false;
        }
        if (firstStep != ExtractionStep::kMapValues) {
          allValues = false;
        }
        if (firstStep != ExtractionStep::kSize) {
          allSize = false;
        }
        if (firstStep != ExtractionStep::kMapKeyFilter) {
          allMapKeyFilter = false;
        }
      }

      if (allSize) {
        spec.setExtractionType(common::ScanSpec::ExtractionType::kSize);
      } else if (allKeys) {
        spec.setExtractionType(common::ScanSpec::ExtractionType::kKeys);
        auto subChains = buildSubChains(extractions, ExtractionStep::kMapKeys);
        if (!subChains.empty()) {
          auto* keysSpec =
              spec.childByName(common::ScanSpec::kMapKeysFieldName);
          if (keysSpec) {
            configureExtractionScanSpec(
                hiveType->asMap().keyType(), subChains, *keysSpec, pool);
          }
        }
      } else if (allValues) {
        spec.setExtractionType(common::ScanSpec::ExtractionType::kValues);
        auto subChains =
            buildSubChains(extractions, ExtractionStep::kMapValues);
        if (!subChains.empty()) {
          auto* valuesSpec =
              spec.childByName(common::ScanSpec::kMapValuesFieldName);
          if (valuesSpec) {
            configureExtractionScanSpec(
                hiveType->asMap().valueType(), subChains, *valuesSpec, pool);
          }
        }
      } else if (allMapKeyFilter) {
        // kMapKeyFilter is type-preserving (MAP -> MAP), so no ExtractionType
        // is set.  Instead, add an IN filter on the map keys ScanSpec so the
        // reader can skip non-matching key-value pairs.
        auto* keysSpec = spec.childByName(common::ScanSpec::kMapKeysFieldName);
        if (keysSpec) {
          // Merge filter keys from all extraction chains.
          std::vector<std::string> mergedStringKeys;
          std::vector<int64_t> mergedIntKeys;
          bool useStringKeys = false;
          for (const auto& extraction : extractions) {
            if (auto* strFilter = dynamic_cast<
                    const StringMapKeyFilterExtractionPathElement*>(
                    extraction.chain[0].get())) {
              useStringKeys = true;
              mergedStringKeys.insert(
                  mergedStringKeys.end(),
                  strFilter->filterKeys().begin(),
                  strFilter->filterKeys().end());
            } else if (
                auto* intFilter =
                    dynamic_cast<const IntMapKeyFilterExtractionPathElement*>(
                        extraction.chain[0].get())) {
              mergedIntKeys.insert(
                  mergedIntKeys.end(),
                  intFilter->filterKeys().begin(),
                  intFilter->filterKeys().end());
            }
          }

          if (useStringKeys) {
            keysSpec->setFilter(
                std::make_unique<common::BytesValues>(
                    mergedStringKeys, /*nullAllowed=*/false));
          } else {
            keysSpec->setFilter(
                common::createBigintValues(
                    mergedIntKeys, /*nullAllowed=*/false));
          }
        }

        // Recurse with remaining chains (strip the kMapKeyFilter step).
        auto subChains =
            buildSubChains(extractions, ExtractionStep::kMapKeyFilter);
        if (!subChains.empty()) {
          configureExtractionScanSpec(hiveType, subChains, spec, pool);
        }
      }
      break;
    }
    case TypeKind::ARRAY: {
      bool allSize = true;
      for (const auto& extraction : extractions) {
        if (extraction.chain.empty() ||
            extraction.chain[0]->step() != ExtractionStep::kSize) {
          allSize = false;
          break;
        }
      }
      if (allSize) {
        spec.setExtractionType(common::ScanSpec::ExtractionType::kSize);
      }
      break;
    }
    case TypeKind::ROW: {
      auto neededFields = analyzeStructNeeds(extractions);
      if (neededFields.empty()) {
        // Need all fields: no pruning.
        break;
      }
      const auto& rowType = hiveType->asRow();

      // If exactly one field is needed, set kField extraction so the struct
      // reader produces the field's vector directly instead of a RowVector.
      if (neededFields.size() == 1) {
        auto& onlyFieldName = *neededFields.begin();
        auto fieldIdx = rowType.getChildIdx(onlyFieldName);
        spec.setExtractionType(common::ScanSpec::ExtractionType::kField);
        spec.setExtractionFieldIndex(fieldIdx);
      }

      for (uint32_t i = 0; i < rowType.size(); ++i) {
        auto& fieldName = rowType.nameOf(i);
        if (neededFields.count(fieldName) == 0) {
          auto* child = spec.childByName(fieldName);
          if (child) {
            child->setConstantValue(
                BaseVector::createNullConstant(rowType.childAt(i), 1, pool));
          }
        } else {
          // Recurse into needed fields with their sub-chains.
          auto subChains = buildSubChains(
              extractions, ExtractionStep::kStructField, fieldName);
          if (!subChains.empty()) {
            auto* child = spec.childByName(fieldName);
            if (child) {
              configureExtractionScanSpec(
                  rowType.childAt(i), subChains, *child, pool);
            }
          }
        }
      }
      break;
    }
    default:
      break;
  }
}

} // namespace facebook::velox::connector::hive
