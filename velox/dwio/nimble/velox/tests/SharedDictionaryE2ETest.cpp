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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryCatalog.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"
#include "velox/dwio/nimble/tablet/SharedDictionaryReader.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/VeloxReader.h"
#include "velox/dwio/nimble/velox/tests/SharedDictionaryTestUtils.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble {
namespace {

std::vector<int32_t> toLogicalValues(const std::vector<uint32_t>& values) {
  std::vector<int32_t> logicalValues;
  logicalValues.reserve(values.size());
  for (const auto value : values) {
    logicalValues.push_back(
        EncodingPhysicalType<int32_t>::asEncodingLogicalType(value));
  }
  return logicalValues;
}

int32_t directStripeValue(velox::vector_size_t row) {
  return row % 2 == 0 ? 10'000 + row
                      : std::numeric_limits<int32_t>::max() - row;
}

static const std::vector<int32_t> kDictionaryStripeAlphabetValues{
    0,
    std::numeric_limits<int32_t>::max()};

std::vector<int32_t> externalDictionaryValues(
    velox::vector_size_t directValueCount) {
  auto values = kDictionaryStripeAlphabetValues;
  values.reserve(values.size() + directValueCount);
  for (velox::vector_size_t row{0}; row < directValueCount; ++row) {
    values.push_back(directStripeValue(row));
  }
  return values;
}

std::vector<int32_t> externalDictionaryValues() {
  return externalDictionaryValues(2'000);
}

enum class InputType {
  FlatMapScalar,
  NullableFlatMapScalar,
  FlatMapArray,
  FlatMapArrayArray,
  FlatMapMap,
  FlatMapRowValue,
  NestedScalar,
  NestedArray,
  NestedMap,
  NestedArrayRow,
  NestedMapRow,
};

enum class StripeValueType {
  Dictionary,
  Direct,
};

enum class DictionaryTargetSet {
  Default,
  FlatMapRowValueOther,
  FlatMapRowValueSubfields,
};

enum class InvalidDictionaryTarget {
  FlatMapWholeRowValue,
  RegularRowColumnValue,
};

enum class ExternalDictionaryFailure {
  MissingResolver,
  MissingValue,
};

enum class FileDictionaryAlphabetOrder {
  Sorted,
  Unsorted,
};

struct TabletReaderDictionaryApiTestCase {
  SharedDictionaryScope scope;
  bool hasStripeDictionaries;
  bool hasFileOrExternalDictionaries;
  bool hasStripeDictionaryStreamId;
  bool resolvesDictionaryAlphabet;
};

std::string_view inputTypeName(InputType inputType) {
  switch (inputType) {
    case InputType::FlatMapScalar:
      return "FlatMapScalar";
    case InputType::NullableFlatMapScalar:
      return "NullableFlatMapScalar";
    case InputType::FlatMapArray:
      return "FlatMapArray";
    case InputType::FlatMapArrayArray:
      return "FlatMapArrayArray";
    case InputType::FlatMapMap:
      return "FlatMapMap";
    case InputType::FlatMapRowValue:
      return "FlatMapRowValue";
    case InputType::NestedScalar:
      return "NestedScalar";
    case InputType::NestedArray:
      return "NestedArray";
    case InputType::NestedMap:
      return "NestedMap";
    case InputType::NestedArrayRow:
      return "NestedArrayRow";
    case InputType::NestedMapRow:
      return "NestedMapRow";
  }
  NIMBLE_UNREACHABLE("Unsupported input type.");
}

std::string_view invalidDictionaryTargetName(InvalidDictionaryTarget target) {
  switch (target) {
    case InvalidDictionaryTarget::FlatMapWholeRowValue:
      return "FlatMapWholeRowValue";
    case InvalidDictionaryTarget::RegularRowColumnValue:
      return "RegularRowColumnValue";
  }
  NIMBLE_UNREACHABLE("Unsupported invalid dictionary target.");
}

std::string_view dictionaryTargetSetName(DictionaryTargetSet targetSet) {
  switch (targetSet) {
    case DictionaryTargetSet::Default:
      return "Default";
    case DictionaryTargetSet::FlatMapRowValueOther:
      return "FlatMapRowValueOther";
    case DictionaryTargetSet::FlatMapRowValueSubfields:
      return "FlatMapRowValueSubfields";
  }
  NIMBLE_UNREACHABLE("Unsupported dictionary target set.");
}

std::string_view externalDictionaryFailureName(
    ExternalDictionaryFailure failure) {
  switch (failure) {
    case ExternalDictionaryFailure::MissingResolver:
      return "MissingResolver";
    case ExternalDictionaryFailure::MissingValue:
      return "MissingValue";
  }
  NIMBLE_UNREACHABLE("Unsupported external dictionary failure.");
}

std::string_view fileDictionaryAlphabetOrderName(
    FileDictionaryAlphabetOrder order) {
  switch (order) {
    case FileDictionaryAlphabetOrder::Sorted:
      return "Sorted";
    case FileDictionaryAlphabetOrder::Unsorted:
      return "Unsorted";
  }
  NIMBLE_UNREACHABLE("Unsupported file dictionary alphabet order.");
}

class FixedFileResolver final : public ExternalDictionaryResolver {
 public:
  FixedFileResolver(
      std::vector<int32_t> values,
      velox::memory::MemoryPool* pool,
      std::span<const EncodingType> candidateEncodings = {})
      : alphabet_{test::createSharedDictionaryAlphabet<int32_t>(
            std::span<const int32_t>{values},
            candidateEncodings,
            pool)} {}

  std::shared_ptr<const SharedDictionaryAlphabet> resolve(
      uint32_t dictionaryId,
      DataType dataType) const final {
    NIMBLE_CHECK_EQ(dictionaryId, 17);
    NIMBLE_CHECK_EQ(dataType, DataType::Int32);
    return alphabet_;
  }

 private:
  std::shared_ptr<const SharedDictionaryAlphabet> alphabet_;
};

class SharedDictionaryE2ETest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::SharedArbitrator::registerFactory();
    velox::memory::MemoryManager::Options options;
    options.arbitratorKind = "SHARED";
    velox::memory::MemoryManager::testingSetInstance(options);
  }

  void SetUp() final {
    rootPool_ = velox::memory::memoryManager()->addRootPool("shared_dict_root");
    leafPool_ = rootPool_->addLeafChild("shared_dict_leaf");
  }

  static void useSharedDictionarySelectionPolicy(WriterOptions& options) {
    test::configureSharedDictionarySelectionPolicy(
        options, {.forceDictionaryForInt32 = false});
  }

  std::shared_ptr<const ExternalDictionaryResolver> makeExternalResolver(
      std::vector<int32_t> values) {
    return makeExternalResolver(
        std::vector<test::SharedDictionaryTestDictionary>{
            {17, std::move(values)}});
  }

  std::shared_ptr<const ExternalDictionaryResolver> makeExternalResolver(
      std::vector<test::SharedDictionaryTestDictionary> dictionaries) {
    return std::make_shared<test::SharedDictionaryTestResolver>(
        std::move(dictionaries), leafPool_.get());
  }

  static constexpr velox::vector_size_t kStripeRows{2'000};

  static std::vector<velox::vector_size_t> makeOffsets(
      velox::vector_size_t rowCount,
      velox::vector_size_t rowWidth = 1) {
    std::vector<velox::vector_size_t> offsets(rowCount);
    for (velox::vector_size_t row{0}; row < rowCount; ++row) {
      offsets[row] = row * rowWidth;
    }
    return offsets;
  }

  static int32_t dictionaryStripeValue(velox::vector_size_t position) {
    return kDictionaryStripeAlphabetValues.at(
        position % kDictionaryStripeAlphabetValues.size());
  }

  static velox::vector_size_t dictionaryValueCount(InputType inputType) {
    switch (inputType) {
      case InputType::FlatMapArray:
      case InputType::FlatMapArrayArray:
      case InputType::FlatMapMap:
      case InputType::FlatMapRowValue:
      case InputType::NestedArray:
      case InputType::NestedArrayRow:
        return kStripeRows * 2;
      case InputType::FlatMapScalar:
      case InputType::NullableFlatMapScalar:
      case InputType::NestedScalar:
      case InputType::NestedMap:
      case InputType::NestedMapRow:
        return kStripeRows;
    }
    NIMBLE_UNREACHABLE("Unsupported input type.");
  }

  static int32_t stripeValue(
      StripeValueType stripeValueType,
      velox::vector_size_t position) {
    if (stripeValueType == StripeValueType::Direct) {
      return directStripeValue(position);
    }
    return dictionaryStripeValue(position);
  }

  velox::RowVectorPtr makeStripe(
      InputType type,
      StripeValueType stripeValueType) {
    velox::test::VectorMaker maker{leafPool_.get()};
    switch (type) {
      case InputType::FlatMapScalar:
        return maker.rowVector(
            {"features"},
            {maker.mapVector<int64_t, int32_t>(
                kStripeRows,
                [](auto /* row */) { return 1; },
                [](auto /* idx */) { return 10; },
                [stripeValueType](auto row) {
                  return stripeValue(stripeValueType, row);
                })});
      case InputType::NullableFlatMapScalar: {
        auto offsets = makeOffsets(kStripeRows);
        auto keys = maker.flatVector<int64_t>(
            kStripeRows, [](auto /*row*/) { return 10; });
        auto values = maker.flatVector<int32_t>(
            kStripeRows,
            [stripeValueType](auto row) {
              return stripeValue(stripeValueType, row);
            },
            [](auto row) { return row % 5 == 0; });
        return maker.rowVector(
            {"features"}, {maker.mapVector(offsets, keys, values)});
      }
      case InputType::FlatMapArray: {
        auto offsets = makeOffsets(kStripeRows);
        auto keys = maker.flatVector<int64_t>(
            kStripeRows, [](auto /*row*/) { return 10; });

        std::vector<std::vector<int32_t>> valueArrays;
        valueArrays.reserve(kStripeRows);
        for (velox::vector_size_t row{0}; row < kStripeRows; ++row) {
          valueArrays.push_back(
              {stripeValue(stripeValueType, row * 2),
               stripeValue(stripeValueType, row * 2 + 1)});
        }

        return maker.rowVector(
            {"features"},
            {maker.mapVector(
                offsets, keys, maker.arrayVector<int32_t>(valueArrays))});
      }
      case InputType::FlatMapArrayArray: {
        auto mapOffsets = makeOffsets(kStripeRows);
        auto keys = maker.flatVector<int64_t>(
            kStripeRows, [](auto /*row*/) { return 10; });
        auto innerArrayOffsets = makeOffsets(kStripeRows, /*rowWidth=*/2);
        auto innerArrays = maker.arrayVector(
            innerArrayOffsets,
            maker.flatVector<int32_t>(
                kStripeRows * 2, [stripeValueType](auto position) {
                  return stripeValue(stripeValueType, position);
                }));
        auto outerArrayOffsets = makeOffsets(kStripeRows);
        return maker.rowVector(
            {"features"},
            {maker.mapVector(
                mapOffsets,
                keys,
                maker.arrayVector(outerArrayOffsets, innerArrays))});
      }
      case InputType::FlatMapMap: {
        auto mapOffsets = makeOffsets(kStripeRows);
        auto keys = maker.flatVector<int64_t>(
            kStripeRows, [](auto /*row*/) { return 10; });
        auto valueMapOffsets = makeOffsets(kStripeRows, /*rowWidth=*/2);
        auto valueMapKeys = maker.flatVector<int64_t>(
            kStripeRows * 2, [](auto position) { return position % 2; });
        auto valueMapValues = maker.flatVector<int32_t>(
            kStripeRows * 2, [stripeValueType](auto position) {
              return stripeValue(stripeValueType, position);
            });
        return maker.rowVector(
            {"features"},
            {maker.mapVector(
                mapOffsets,
                keys,
                maker.mapVector(
                    valueMapOffsets, valueMapKeys, valueMapValues))});
      }
      case InputType::FlatMapRowValue: {
        auto mapOffsets = makeOffsets(kStripeRows);
        auto keys = maker.flatVector<int64_t>(
            kStripeRows, [](auto /*row*/) { return 10; });
        auto arrayOffsets = makeOffsets(kStripeRows, /*rowWidth=*/2);
        auto values = maker.rowVector(
            {"items", "other"},
            {
                maker.arrayVector(
                    arrayOffsets,
                    maker.flatVector<int32_t>(
                        kStripeRows * 2,
                        [stripeValueType](auto position) {
                          return stripeValue(stripeValueType, position);
                        })),
                maker.flatVector<int32_t>(
                    kStripeRows,
                    [stripeValueType](auto row) {
                      return stripeValue(stripeValueType, row);
                    }),
            });
        return maker.rowVector(
            {"features"}, {maker.mapVector(mapOffsets, keys, values)});
      }
      case InputType::NestedScalar:
        return maker.rowVector(
            {"nested"},
            {maker.rowVector(
                {"value"},
                {maker.flatVector<int32_t>(
                    kStripeRows, [stripeValueType](auto row) {
                      return stripeValue(stripeValueType, row);
                    })})});
      case InputType::NestedArray: {
        std::vector<std::vector<int32_t>> valueArrays;
        valueArrays.reserve(kStripeRows);
        for (velox::vector_size_t row{0}; row < kStripeRows; ++row) {
          valueArrays.push_back(
              {stripeValue(stripeValueType, row * 2),
               stripeValue(stripeValueType, row * 2 + 1)});
        }
        return maker.rowVector(
            {"nested"},
            {maker.rowVector(
                {"items"}, {maker.arrayVector<int32_t>(valueArrays)})});
      }
      case InputType::NestedMap: {
        auto offsets = makeOffsets(kStripeRows);
        auto keys = maker.flatVector<int64_t>(
            kStripeRows, [](auto row) { return row + 100; });
        auto values =
            maker.flatVector<int32_t>(kStripeRows, [stripeValueType](auto row) {
              return stripeValue(stripeValueType, row);
            });
        return maker.rowVector(
            {"nested"},
            {maker.rowVector(
                {"props"}, {maker.mapVector(offsets, keys, values)})});
      }
      case InputType::NestedArrayRow: {
        auto offsets = makeOffsets(kStripeRows, /*rowWidth=*/2);
        auto elements = maker.rowVector(
            {"value", "other"},
            {
                maker.flatVector<int32_t>(
                    kStripeRows * 2,
                    [stripeValueType](auto position) {
                      return stripeValue(stripeValueType, position);
                    }),
                maker.flatVector<int32_t>(
                    kStripeRows * 2,
                    [](auto position) { return directStripeValue(position); }),
            });
        return maker.rowVector(
            {"nested"},
            {maker.rowVector(
                {"items"}, {maker.arrayVector(offsets, elements)})});
      }
      case InputType::NestedMapRow: {
        auto offsets = makeOffsets(kStripeRows);
        auto keys = maker.flatVector<int64_t>(
            kStripeRows, [](auto row) { return row + 100; });
        auto values = maker.rowVector(
            {"value", "other"},
            {
                maker.flatVector<int32_t>(
                    kStripeRows,
                    [stripeValueType](auto row) {
                      return stripeValue(stripeValueType, row);
                    }),
                maker.flatVector<int32_t>(
                    kStripeRows,
                    [](auto row) { return directStripeValue(row); }),
            });
        return maker.rowVector(
            {"nested"},
            {maker.rowVector(
                {"props"}, {maker.mapVector(offsets, keys, values)})});
      }
    }
    NIMBLE_UNREACHABLE("Unsupported stripe type.");
  }

  velox::RowVectorPtr makeDictionaryStripe(InputType type) {
    return makeStripe(type, StripeValueType::Dictionary);
  }

  velox::RowVectorPtr makeDirectStripe(InputType type) {
    return makeStripe(type, StripeValueType::Direct);
  }

  velox::RowVectorPtr makeFileDictionaryStripe(
      FileDictionaryAlphabetOrder alphabetOrder) {
    velox::test::VectorMaker maker{leafPool_.get()};
    const auto values = fileDictionaryValues(alphabetOrder);
    return maker.rowVector(
        {"features"},
        {maker.mapVector<int64_t, int32_t>(
            2'000,
            [](auto /* row */) { return 1; },
            [](auto /* idx */) { return 10; },
            [&](auto idx) { return values[idx % values.size()]; })});
  }

  static std::vector<int32_t> fileDictionaryValues(
      FileDictionaryAlphabetOrder order) {
    switch (order) {
      case FileDictionaryAlphabetOrder::Sorted:
        return {10, 20, 30, 40};
      case FileDictionaryAlphabetOrder::Unsorted:
        return {40, 10, 30, 20};
    }
    NIMBLE_UNREACHABLE("Unsupported file dictionary alphabet order.");
  }

  void addFlatmapDictionary(
      WriterOptions& options,
      SharedDictionaryConfig dictionary) {
    options.experimentalSharedDictionaryEncoding =
        SharedDictionaryEncodingConfig::builder(
            std::move(options.experimentalSharedDictionaryEncoding))
            .addFlatmapValueDictionary(
                "features",
                /*key=*/10,
                std::move(dictionary))
            .build();
  }

  void addFlatmapDictionary(
      WriterOptions& options,
      SharedDictionaryConfig dictionary,
      std::string valueSubfield) {
    options.experimentalSharedDictionaryEncoding =
        SharedDictionaryEncodingConfig::builder(
            std::move(options.experimentalSharedDictionaryEncoding))
            .addFlatmapValueDictionary(
                "features",
                /*key=*/10,
                std::move(dictionary),
                std::move(valueSubfield))
            .build();
  }

  void addColumnDictionary(
      WriterOptions& options,
      SharedDictionaryConfig dictionary,
      std::string fieldPath) {
    options.experimentalSharedDictionaryEncoding =
        SharedDictionaryEncodingConfig::builder(
            std::move(options.experimentalSharedDictionaryEncoding))
            .addColumnDictionary(std::move(fieldPath), std::move(dictionary))
            .build();
  }

  static SharedDictionaryConfig sharedDictionaryConfig(
      SharedDictionaryScope scope,
      uint32_t dictionaryId) {
    SharedDictionaryConfig dictionary{.scope = scope};
    if (scope != SharedDictionaryScope::Stripe) {
      dictionary.dictionaryId = dictionaryId;
    }
    return dictionary;
  }

  void addDictionary(
      WriterOptions& options,
      InputType inputType,
      SharedDictionaryConfig dictionary) {
    switch (inputType) {
      case InputType::FlatMapScalar:
      case InputType::NullableFlatMapScalar:
      case InputType::FlatMapArray:
      case InputType::FlatMapArrayArray:
      case InputType::FlatMapMap:
        options.flatMapColumns.emplace("features", std::set<std::string>{});
        addFlatmapDictionary(options, std::move(dictionary));
        return;
      case InputType::FlatMapRowValue:
        options.flatMapColumns.emplace("features", std::set<std::string>{});
        addFlatmapDictionary(
            options, std::move(dictionary), /*valueSubfield=*/"items[*]");
        return;
      case InputType::NestedScalar:
        addColumnDictionary(options, std::move(dictionary), "nested.value");
        return;
      case InputType::NestedArray:
        addColumnDictionary(options, std::move(dictionary), "nested.items");
        return;
      case InputType::NestedMap:
        addColumnDictionary(options, std::move(dictionary), "nested.props");
        return;
      case InputType::NestedArrayRow:
        addColumnDictionary(
            options, std::move(dictionary), "nested.items[*].value");
        return;
      case InputType::NestedMapRow:
        addColumnDictionary(
            options, std::move(dictionary), "nested.props[*].value");
        return;
    }
    NIMBLE_UNREACHABLE("Unsupported input type.");
  }

  void addFlatMapRowValueSubfieldDictionaries(
      WriterOptions& options,
      SharedDictionaryScope scope) {
    options.flatMapColumns.emplace("features", std::set<std::string>{});
    addFlatmapDictionary(
        options,
        sharedDictionaryConfig(scope, /*dictionaryId=*/17),
        "items[*]");
    addFlatmapDictionary(
        options, sharedDictionaryConfig(scope, /*dictionaryId=*/18), "other");
  }

  void addFlatMapRowValueOtherDictionary(
      WriterOptions& options,
      SharedDictionaryScope scope) {
    options.flatMapColumns.emplace("features", std::set<std::string>{});
    addFlatmapDictionary(
        options,
        sharedDictionaryConfig(scope, /*dictionaryId=*/17),
        /*valueSubfield=*/"other");
  }

  WriterOptions makeSharedDictionaryWriterOptions() {
    WriterOptions options;
    options.maxStreamChunkRawSize = 512;
    options.minStreamChunkRawSize = 1;
    useSharedDictionarySelectionPolicy(options);
    return options;
  }

  std::string writeInput(
      const std::vector<velox::RowVectorPtr>& stripeInputs,
      WriterOptions options) {
    NIMBLE_CHECK(!stripeInputs.empty(), "Must write at least one stripe.");
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    Writer writer{
        stripeInputs.front()->type(),
        std::move(writeFile),
        *rootPool_,
        std::move(options)};
    for (size_t stripeIndex{0}; stripeIndex < stripeInputs.size();
         ++stripeIndex) {
      if (stripeIndex > 0) {
        writer.flush();
      }
      writer.write(stripeInputs[stripeIndex]);
    }
    writer.close();
    return file;
  }

  std::string write(
      InputType inputType,
      SharedDictionaryScope scope,
      std::vector<StripeValueType> stripeValueTypes =
          {StripeValueType::Dictionary, StripeValueType::Direct},
      std::shared_ptr<const ExternalDictionaryResolver>
          externalDictionaryResolver = nullptr,
      DictionaryTargetSet dictionaryTargetSet = DictionaryTargetSet::Default) {
    NIMBLE_CHECK(!stripeValueTypes.empty(), "Must write at least one stripe.");
    auto options = makeSharedDictionaryWriterOptions();
    if (dictionaryTargetSet == DictionaryTargetSet::FlatMapRowValueSubfields) {
      NIMBLE_CHECK(
          inputType == InputType::FlatMapRowValue,
          "FlatMapRowValueSubfields only applies to FlatMapRowValue input.");
      addFlatMapRowValueSubfieldDictionaries(options, scope);
    } else if (
        dictionaryTargetSet == DictionaryTargetSet::FlatMapRowValueOther) {
      NIMBLE_CHECK(
          inputType == InputType::FlatMapRowValue,
          "FlatMapRowValueOther only applies to FlatMapRowValue input.");
      addFlatMapRowValueOtherDictionary(options, scope);
    } else {
      addDictionary(
          options,
          inputType,
          sharedDictionaryConfig(
              scope, scope == SharedDictionaryScope::External ? 17 : 7));
    }
    options.experimentalSharedDictionaryEncoding.externalResolver =
        std::move(externalDictionaryResolver);

    std::vector<velox::RowVectorPtr> stripeInputs;
    stripeInputs.reserve(stripeValueTypes.size());
    for (const auto stripeValueType : stripeValueTypes) {
      stripeInputs.push_back(makeStripe(inputType, stripeValueType));
    }
    return writeInput(stripeInputs, std::move(options));
  }

  std::shared_ptr<const ExternalDictionaryResolver> makeExternalResolver(
      InputType inputType,
      DictionaryTargetSet dictionaryTargetSet = DictionaryTargetSet::Default) {
    switch (dictionaryTargetSet) {
      case DictionaryTargetSet::Default:
        return makeExternalResolver(
            externalDictionaryValues(dictionaryValueCount(inputType)));
      case DictionaryTargetSet::FlatMapRowValueOther:
        NIMBLE_CHECK(
            inputType == InputType::FlatMapRowValue,
            "FlatMapRowValueOther only applies to FlatMapRowValue input.");
        return makeExternalResolver(externalDictionaryValues(kStripeRows));
      case DictionaryTargetSet::FlatMapRowValueSubfields:
        NIMBLE_CHECK(
            inputType == InputType::FlatMapRowValue,
            "FlatMapRowValueSubfields only applies to FlatMapRowValue input.");
        return makeExternalResolver(
            std::vector<test::SharedDictionaryTestDictionary>{
                {17, externalDictionaryValues(kStripeRows * 2)},
                {18, externalDictionaryValues()}});
    }
    NIMBLE_UNREACHABLE("Unsupported dictionary target set.");
  }

  static DictionaryTargetSet randomDictionaryTargetSet(
      InputType inputType,
      std::mt19937& rng) {
    if (inputType != InputType::FlatMapRowValue) {
      return DictionaryTargetSet::Default;
    }
    const std::array<DictionaryTargetSet, 3> targetSets{
        DictionaryTargetSet::Default,
        DictionaryTargetSet::FlatMapRowValueOther,
        DictionaryTargetSet::FlatMapRowValueSubfields};
    return targetSets[std::uniform_int_distribution<size_t>{
        0, targetSets.size() - 1}(rng)];
  }

  static size_t expectedSharedDictionaryValueStreamCount(
      DictionaryTargetSet dictionaryTargetSet) {
    return dictionaryTargetSet == DictionaryTargetSet::FlatMapRowValueSubfields
        ? 2
        : 1;
  }

  std::string writeWithCompressedFileDictionary(size_t stripeCount) {
    auto input = makeDictionaryStripe(InputType::FlatMapScalar);
    auto options = makeSharedDictionaryWriterOptions();
    options.enableChunking = true;
    options.metadataCompressionThreshold = std::numeric_limits<uint32_t>::max();
    options.chunkCompression = {
        .type = CompressionType::Zstd, .zstdLevel = 3, .acceptRatio = 1.0f};
    addDictionary(
        options,
        InputType::FlatMapScalar,
        SharedDictionaryConfig{
            .scope = SharedDictionaryScope::File, .dictionaryId = 17});

    return writeInput(
        std::vector<velox::RowVectorPtr>(stripeCount, input),
        std::move(options));
  }

  std::string writeWithFileDictionary(
      size_t stripeCount,
      FileDictionaryAlphabetOrder alphabetOrder) {
    auto input = makeFileDictionaryStripe(alphabetOrder);
    auto resolver = std::make_shared<FixedFileResolver>(
        fileDictionaryValues(alphabetOrder),
        leafPool_.get(),
        std::array{EncodingType::FixedBitWidth});
    auto options = makeSharedDictionaryWriterOptions();
    options.experimentalSharedDictionaryEncoding.externalResolver =
        std::move(resolver);
    addDictionary(
        options,
        InputType::FlatMapScalar,
        SharedDictionaryConfig{
            .scope = SharedDictionaryScope::File,
            .dictionaryId = 17,
            .useExternalAlphabet = true,
            // Ignored for prebuilt external alphabets; the resolver's
            // FixedBitWidth encoding should be preserved instead.
            .alphabetEncodings = {EncodingType::DeltaBlock}});

    return writeInput(
        std::vector<velox::RowVectorPtr>(stripeCount, input),
        std::move(options));
  }

  std::shared_ptr<TabletReader> openTablet(
      const std::string& file,
      std::shared_ptr<const ExternalDictionaryResolver> externalResolver =
          nullptr) {
    auto options = test::makeTestTabletOptions(leafPool_.get());
    options.externalDictionaryResolver = std::move(externalResolver);
    return TabletReader::create(
        std::make_shared<velox::InMemoryReadFile>(file),
        leafPool_.get(),
        options);
  }

  std::shared_ptr<const Type> readSchema(const TabletReader& tablet) {
    auto schemaSection =
        tablet.loadOptionalSection(std::string(kSchemaSection));
    NIMBLE_CHECK(schemaSection.has_value());
    return SchemaDeserializer::deserialize(schemaSection->content().data());
  }

  const Type& flatMapValueType(
      const Type& schema,
      std::string_view key = "10") {
    const auto& flatMap = schema.asRow().childAt(0)->asFlatMap();
    auto child = flatMap.findChild(key);
    NIMBLE_CHECK(child.has_value());
    return *flatMap.childAt(child.value());
  }

  uint32_t scalarStreamId(const Type& type) {
    return type.asScalar().scalarDescriptor().offset();
  }

  std::vector<uint32_t> scalarStreamIds(const Type& type) {
    if (type.isScalar()) {
      return {scalarStreamId(type)};
    }
    if (type.isArray()) {
      return scalarStreamIds(*type.asArray().elements());
    }
    if (type.isMap()) {
      return scalarStreamIds(*type.asMap().values());
    }
    if (type.isRow()) {
      const auto& rowType = type.asRow();
      std::vector<uint32_t> streamIds;
      for (size_t childIndex{0}; childIndex < rowType.childrenCount();
           ++childIndex) {
        const auto childStreamIds =
            scalarStreamIds(*rowType.childAt(childIndex));
        streamIds.insert(
            streamIds.end(), childStreamIds.begin(), childStreamIds.end());
      }
      return streamIds;
    }
    NIMBLE_UNREACHABLE("Unsupported dictionary value type.");
  }

  std::vector<uint32_t> candidateDictionaryValueStreamIds(
      const TabletReader& tablet,
      InputType inputType) {
    const auto schema = readSchema(tablet);
    switch (inputType) {
      case InputType::FlatMapScalar:
      case InputType::NullableFlatMapScalar:
      case InputType::FlatMapArray:
      case InputType::FlatMapArrayArray:
      case InputType::FlatMapMap:
      case InputType::FlatMapRowValue:
        return scalarStreamIds(flatMapValueType(*schema));
      case InputType::NestedScalar:
      case InputType::NestedArray:
      case InputType::NestedMap:
      case InputType::NestedArrayRow:
      case InputType::NestedMapRow:
        return scalarStreamIds(*schema->asRow().childAt(0));
    }
    NIMBLE_UNREACHABLE("Unsupported input type.");
  }

  bool hasSharedDictionary(const TabletReader& tablet, uint32_t valueStreamId) {
    return tablet.stripeDictionaryStreamId(valueStreamId).has_value() ||
        tablet.resolveDictionaryAlphabet(valueStreamId) != nullptr;
  }

  enum class EncodingTypeComparison {
    Equal,
    NotEqual,
  };

  using EncodingTypesByStripe = std::vector<std::vector<EncodingType>>;

  std::vector<uint32_t> sharedDictionaryValueStreamIds(
      const TabletReader& tablet,
      InputType inputType) {
    std::vector<uint32_t> streamIds;
    for (const auto streamId :
         candidateDictionaryValueStreamIds(tablet, inputType)) {
      if (hasSharedDictionary(tablet, streamId)) {
        streamIds.push_back(streamId);
      }
    }
    NIMBLE_CHECK(
        !streamIds.empty(),
        "Expected at least one shared dictionary value stream.");
    return streamIds;
  }

  EncodingType valueEncodingType(std::string_view chunk) {
    const auto encodingType = EncodingPrefix::encodingType(chunk);
    if (encodingType != EncodingType::Nullable) {
      return encodingType;
    }

    const char* dataChild =
        chunk.data() + EncodingPrefix::prefixSize(chunk, /*useVarint=*/false);
    const auto dataChildSize = encoding::readUint32(dataChild);
    return EncodingPrefix::encodingType({dataChild, dataChildSize});
  }

  template <typename EncodingTypeReader>
  EncodingTypesByStripe collectStreamEncodingTypesByStripe(
      const TabletReader& tablet,
      uint32_t streamId,
      EncodingTypeReader readEncodingType) {
    EncodingTypesByStripe typesByStripe;
    typesByStripe.reserve(tablet.stripeCount());
    for (uint32_t stripeIndex{0}; stripeIndex < tablet.stripeCount();
         ++stripeIndex) {
      const std::array<uint32_t, 1> ids{streamId};
      auto streams = tablet.load(tablet.stripeIdentifier(stripeIndex), ids);
      NIMBLE_CHECK_EQ(streams.size(), 1);
      NIMBLE_CHECK_NOT_NULL(streams.front().get());
      InMemoryChunkedStream chunks{*leafPool_, std::move(streams.front())};
      auto& stripeTypes = typesByStripe.emplace_back();
      while (chunks.hasNext()) {
        stripeTypes.push_back(readEncodingType(chunks.nextChunk()));
      }
      NIMBLE_CHECK(!stripeTypes.empty());
    }
    return typesByStripe;
  }

  EncodingTypesByStripe streamEncodingTypesByStripe(
      const TabletReader& tablet,
      uint32_t streamId) {
    return collectStreamEncodingTypesByStripe(
        tablet, streamId, [](std::string_view chunk) {
          return EncodingPrefix::encodingType(chunk);
        });
  }

  EncodingTypesByStripe streamDataEncodingTypesByStripe(
      const TabletReader& tablet,
      uint32_t streamId) {
    return collectStreamEncodingTypesByStripe(
        tablet, streamId, [this](std::string_view chunk) {
          return valueEncodingType(chunk);
        });
  }

  void expectStripeEncodingTypes(
      const EncodingTypesByStripe& typesByStripe,
      size_t stripeIndex,
      EncodingType expected,
      EncodingTypeComparison comparison = EncodingTypeComparison::Equal) {
    ASSERT_LT(stripeIndex, typesByStripe.size());
    ASSERT_FALSE(typesByStripe[stripeIndex].empty());
    for (size_t chunkIndex{0}; chunkIndex < typesByStripe[stripeIndex].size();
         ++chunkIndex) {
      SCOPED_TRACE(
          fmt::format(
              "stripeIndex={}, chunkIndex={}", stripeIndex, chunkIndex));
      if (comparison == EncodingTypeComparison::Equal) {
        EXPECT_EQ(typesByStripe[stripeIndex][chunkIndex], expected);
      } else {
        EXPECT_NE(typesByStripe[stripeIndex][chunkIndex], expected);
      }
    }
  }

  void expectAllStripeEncodingTypes(
      const EncodingTypesByStripe& typesByStripe,
      EncodingType expected) {
    ASSERT_FALSE(typesByStripe.empty());
    for (size_t stripeIndex{0}; stripeIndex < typesByStripe.size();
         ++stripeIndex) {
      SCOPED_TRACE(fmt::format("stripeIndex={}", stripeIndex));
      expectStripeEncodingTypes(typesByStripe, stripeIndex, expected);
    }
  }

  void expectMultipleChunksPerStripe(
      const EncodingTypesByStripe& typesByStripe) {
    ASSERT_FALSE(typesByStripe.empty());
    for (size_t stripeIndex{0}; stripeIndex < typesByStripe.size();
         ++stripeIndex) {
      SCOPED_TRACE(fmt::format("stripeIndex={}", stripeIndex));
      EXPECT_GT(typesByStripe[stripeIndex].size(), 1);
    }
  }

  void expectStripeDictionaryStreamPresence(
      const TabletReader& tablet,
      uint32_t valueStreamId,
      std::span<const StripeValueType> stripeValueTypes) {
    ASSERT_EQ(tablet.stripeCount(), stripeValueTypes.size());
    const auto dictionaryStreamId =
        tablet.stripeDictionaryStreamId(valueStreamId);
    ASSERT_TRUE(dictionaryStreamId.has_value());
    const std::array<uint32_t, 1> dictionaryStreamIds{
        dictionaryStreamId.value()};
    for (size_t stripeIndex{0}; stripeIndex < stripeValueTypes.size();
         ++stripeIndex) {
      SCOPED_TRACE(fmt::format("stripeIndex={}", stripeIndex));
      auto dictionaryStreams = tablet.load(
          tablet.stripeIdentifier(stripeIndex), dictionaryStreamIds);
      ASSERT_EQ(dictionaryStreams.size(), 1);
      if (stripeValueTypes[stripeIndex] == StripeValueType::Dictionary) {
        EXPECT_NE(dictionaryStreams.front(), nullptr);
      } else {
        EXPECT_EQ(dictionaryStreams.front(), nullptr);
      }
    }
  }

  void expectDictionaryValueEncodingTypes(
      const TabletReader& tablet,
      uint32_t valueStreamId,
      SharedDictionaryScope scope,
      std::span<const StripeValueType> stripeValueTypes) {
    const auto dataEncodingTypesByStripe =
        streamDataEncodingTypesByStripe(tablet, valueStreamId);
    ASSERT_EQ(dataEncodingTypesByStripe.size(), stripeValueTypes.size());
    if (scope == SharedDictionaryScope::Stripe) {
      expectMultipleChunksPerStripe(dataEncodingTypesByStripe);
      for (size_t stripeIndex{0}; stripeIndex < stripeValueTypes.size();
           ++stripeIndex) {
        SCOPED_TRACE(fmt::format("stripeIndex={}", stripeIndex));
        expectStripeEncodingTypes(
            dataEncodingTypesByStripe,
            stripeIndex,
            EncodingType::SharedDictionary,
            stripeValueTypes[stripeIndex] == StripeValueType::Dictionary
                ? EncodingTypeComparison::Equal
                : EncodingTypeComparison::NotEqual);
      }
      expectStripeDictionaryStreamPresence(
          tablet, valueStreamId, stripeValueTypes);
      return;
    }
    expectAllStripeEncodingTypes(
        dataEncodingTypesByStripe, EncodingType::SharedDictionary);
  }

  CompressionType dictionaryValueChunkCompressionType(
      const std::string& file,
      InputType inputType,
      uint32_t stripeIndex) {
    auto tablet = openTablet(file);
    const auto streamIds = sharedDictionaryValueStreamIds(*tablet, inputType);
    NIMBLE_CHECK_EQ(streamIds.size(), 1);

    const std::array<uint32_t, 1> ids{streamIds.front()};
    auto streams = tablet->load(tablet->stripeIdentifier(stripeIndex), ids);
    InMemoryChunkedStream chunks{*leafPool_, std::move(streams.front())};
    NIMBLE_CHECK(chunks.hasNext());
    return chunks.peekCompressionType();
  }

  EncodingType fileDictionaryAlphabetEncodingType(const std::string& file) {
    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    auto options = test::makeTestTabletOptions(leafPool_.get());
    auto tablet = TabletReader::create(readFile, leafPool_.get(), options);
    auto section = tablet->loadOptionalSection(std::string(kDictionarySection));
    NIMBLE_CHECK(section.has_value());

    const auto catalog =
        SharedDictionaryCatalog::deserialize(section->content());
    const auto& fileDictionaries = catalog.fileDictionaries();
    NIMBLE_CHECK_EQ(fileDictionaries.size(), 1);
    NIMBLE_CHECK_EQ(fileDictionaries[0].dictionaryId, 17);
    NIMBLE_CHECK_EQ(fileDictionaries[0].dataType, DataType::Int32);
    const auto encoded = std::string_view{file}.substr(
        fileDictionaries[0].offset, fileDictionaries[0].length);
    NIMBLE_CHECK_GT(encoded.size(), EncodingPrefix::kEncodingTypeOffset);
    return static_cast<EncodingType>(
        encoded[EncodingPrefix::kEncodingTypeOffset]);
  }

  std::vector<int32_t> fileDictionaryValues(const std::string& file) {
    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    auto options = test::makeTestTabletOptions(leafPool_.get());
    auto tablet = TabletReader::create(readFile, leafPool_.get(), options);
    auto section = tablet->loadOptionalSection(std::string(kDictionarySection));
    NIMBLE_CHECK(section.has_value());
    const auto catalog =
        SharedDictionaryCatalog::deserialize(section->content());
    NIMBLE_CHECK_EQ(catalog.fileDictionaries().size(), 1);
    const auto& fileDictionary = catalog.fileDictionaries()[0];
    auto encodedAlphabetOwner =
        std::make_shared<const std::string>(std::string_view{file}.substr(
            fileDictionary.offset, fileDictionary.length));
    const std::string_view encodedAlphabet{*encodedAlphabetOwner};
    auto alphabet = SharedDictionaryAlphabet::create(
        encodedAlphabet, std::move(encodedAlphabetOwner), leafPool_.get());
    NIMBLE_CHECK_NOT_NULL(alphabet);
    NIMBLE_CHECK_EQ(alphabet->dataType(), fileDictionary.dataType);
    std::vector<uint32_t> physicalValues(alphabet->entryCount());
    std::vector<uint32_t> indices;
    indices.reserve(alphabet->entryCount());
    for (uint32_t i = 0; i < alphabet->entryCount(); ++i) {
      indices.push_back(i);
    }
    alphabet->materialize<int32_t>(indices, physicalValues.data());
    return toLogicalValues(physicalValues);
  }

  std::vector<int32_t> materializedAlphabetValues(
      const SharedDictionaryAlphabet& alphabet) {
    Vector<int32_t> values{leafPool_.get()};
    alphabet.materializeAll<int32_t>(values);
    return {values.begin(), values.end()};
  }

  std::vector<int32_t> expectedResolvedAlphabetValues(
      const std::string& file,
      SharedDictionaryScope scope) {
    switch (scope) {
      case SharedDictionaryScope::File:
        return fileDictionaryValues(file);
      case SharedDictionaryScope::External:
        return externalDictionaryValues();
      case SharedDictionaryScope::Stripe:
        NIMBLE_UNREACHABLE("Stripe dictionaries do not resolve an alphabet.");
    }
    NIMBLE_UNREACHABLE("Unsupported shared dictionary scope.");
  }

  void verifyRoundTrip(
      const std::string& file,
      InputType inputType = InputType::FlatMapScalar,
      std::vector<StripeValueType> stripeValueTypes =
          {StripeValueType::Dictionary, StripeValueType::Direct},
      std::shared_ptr<const ExternalDictionaryResolver> externalResolver =
          nullptr) {
    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    VeloxReadParams params;
    params.externalDictionaryResolver = std::move(externalResolver);
    VeloxReader reader{readFile, *leafPool_, nullptr, params};
    velox::VectorPtr output;
    for (const auto stripeValueType : stripeValueTypes) {
      ASSERT_TRUE(reader.next(kStripeRows, output));
      auto expected = makeStripe(inputType, stripeValueType);
      ASSERT_EQ(output->size(), expected->size());
      for (velox::vector_size_t i = 0; i < output->size(); ++i) {
        ASSERT_TRUE(output->equalValueAt(expected.get(), i, i));
      }
    }
    EXPECT_FALSE(reader.next(kStripeRows, output));
  }

  void verifyDictionaryStripeRoundTrip(
      const std::string& file,
      InputType inputType) {
    verifyRoundTrip(file, inputType, {StripeValueType::Dictionary});
  }

  void verifyFileDictionaryRoundTrip(
      const std::string& file,
      FileDictionaryAlphabetOrder alphabetOrder,
      size_t stripeCount = 1) {
    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    VeloxReader reader{readFile, *leafPool_};
    velox::VectorPtr output;
    auto expected = makeFileDictionaryStripe(alphabetOrder);
    for (size_t stripeIndex{0}; stripeIndex < stripeCount; ++stripeIndex) {
      SCOPED_TRACE(fmt::format("stripeIndex={}", stripeIndex));
      ASSERT_TRUE(reader.next(kStripeRows, output));
      ASSERT_EQ(output->size(), expected->size());
      for (velox::vector_size_t i = 0; i < output->size(); ++i) {
        ASSERT_TRUE(output->equalValueAt(expected.get(), i, i));
      }
    }
    EXPECT_FALSE(reader.next(kStripeRows, output));
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

class SharedDictionaryE2EInputTypeTest
    : public SharedDictionaryE2ETest,
      public testing::WithParamInterface<InputType> {};

class SharedDictionaryE2EFlatMapRowValueSubfieldsTest
    : public SharedDictionaryE2ETest,
      public testing::WithParamInterface<SharedDictionaryScope> {};

class SharedDictionaryE2EFlatMapDefaultValueSubfieldTest
    : public SharedDictionaryE2ETest,
      public testing::WithParamInterface<InputType> {};

class SharedDictionaryE2ETabletReaderApiTest
    : public SharedDictionaryE2ETest,
      public testing::WithParamInterface<TabletReaderDictionaryApiTestCase> {};

TEST_P(SharedDictionaryE2EInputTypeTest, stripeScopeRoundTrip) {
  const auto inputType = GetParam();
  const std::vector<StripeValueType> stripeValueTypes{
      StripeValueType::Dictionary, StripeValueType::Direct};
  const auto file =
      write(inputType, SharedDictionaryScope::Stripe, stripeValueTypes);
  auto tablet = openTablet(file);
  const auto valueStreamIds =
      sharedDictionaryValueStreamIds(*tablet, inputType);
  ASSERT_FALSE(valueStreamIds.empty());
  for (const auto valueStreamId : valueStreamIds) {
    SCOPED_TRACE(fmt::format("valueStreamId={}", valueStreamId));
    expectDictionaryValueEncodingTypes(
        *tablet,
        valueStreamId,
        SharedDictionaryScope::Stripe,
        stripeValueTypes);
  }
  verifyRoundTrip(file, inputType);
}

TEST_P(SharedDictionaryE2EInputTypeTest, fileScopeRoundTrip) {
  const auto inputType = GetParam();
  const std::vector<StripeValueType> stripeValueTypes{
      StripeValueType::Dictionary, StripeValueType::Direct};
  const auto file =
      write(inputType, SharedDictionaryScope::File, stripeValueTypes);
  auto tablet = openTablet(file);
  const auto valueStreamIds =
      sharedDictionaryValueStreamIds(*tablet, inputType);
  ASSERT_EQ(valueStreamIds.size(), 1);
  for (const auto valueStreamId : valueStreamIds) {
    SCOPED_TRACE(fmt::format("valueStreamId={}", valueStreamId));
    expectDictionaryValueEncodingTypes(
        *tablet, valueStreamId, SharedDictionaryScope::File, stripeValueTypes);
    if (inputType == InputType::NullableFlatMapScalar) {
      const auto encodingTypesByStripe =
          streamEncodingTypesByStripe(*tablet, valueStreamId);
      ASSERT_EQ(encodingTypesByStripe.size(), stripeValueTypes.size());
      expectAllStripeEncodingTypes(
          encodingTypesByStripe, EncodingType::Nullable);
    }
  }
  verifyRoundTrip(file, inputType, stripeValueTypes);
}

TEST_P(SharedDictionaryE2EFlatMapRowValueSubfieldsTest, roundTrip) {
  const auto scope = GetParam();
  const std::vector<StripeValueType> stripeValueTypes{
      StripeValueType::Dictionary, StripeValueType::Direct};
  std::shared_ptr<const ExternalDictionaryResolver> resolver;
  if (scope == SharedDictionaryScope::External) {
    resolver = makeExternalResolver(
        std::vector<test::SharedDictionaryTestDictionary>{
            {17, externalDictionaryValues(kStripeRows * 2)},
            {18, externalDictionaryValues()}});
  }
  const auto file = write(
      InputType::FlatMapRowValue,
      scope,
      stripeValueTypes,
      resolver,
      DictionaryTargetSet::FlatMapRowValueSubfields);
  auto tablet = openTablet(file, resolver);
  const auto valueStreamIds =
      sharedDictionaryValueStreamIds(*tablet, InputType::FlatMapRowValue);
  ASSERT_EQ(valueStreamIds.size(), 2);
  for (const auto valueStreamId : valueStreamIds) {
    SCOPED_TRACE(fmt::format("valueStreamId={}", valueStreamId));
    expectDictionaryValueEncodingTypes(
        *tablet, valueStreamId, scope, stripeValueTypes);
  }
  verifyRoundTrip(file, InputType::FlatMapRowValue, stripeValueTypes, resolver);
}

TEST_P(
    SharedDictionaryE2EFlatMapDefaultValueSubfieldTest,
    selectsNestedValueStream) {
  const auto inputType = GetParam();
  const std::vector<StripeValueType> stripeValueTypes{
      StripeValueType::Dictionary, StripeValueType::Direct};
  const auto file =
      write(inputType, SharedDictionaryScope::File, stripeValueTypes);
  auto tablet = openTablet(file);
  const auto expectedValueStreamIds =
      candidateDictionaryValueStreamIds(*tablet, inputType);
  ASSERT_EQ(expectedValueStreamIds.size(), 1);
  const auto valueStreamIds =
      sharedDictionaryValueStreamIds(*tablet, inputType);
  EXPECT_EQ(valueStreamIds, expectedValueStreamIds);
  expectDictionaryValueEncodingTypes(
      *tablet,
      valueStreamIds.front(),
      SharedDictionaryScope::File,
      stripeValueTypes);
  verifyRoundTrip(file, inputType, stripeValueTypes);
}

TEST_F(SharedDictionaryE2ETest, fuzzRoundTrip) {
  constexpr uint32_t kSeed{0x5D1C710A};
  constexpr size_t kIterationCount{50};
  const std::array<InputType, 11> inputTypes{
      InputType::FlatMapScalar,
      InputType::NullableFlatMapScalar,
      InputType::FlatMapArray,
      InputType::FlatMapArrayArray,
      InputType::FlatMapMap,
      InputType::FlatMapRowValue,
      InputType::NestedScalar,
      InputType::NestedArray,
      InputType::NestedMap,
      InputType::NestedArrayRow,
      InputType::NestedMapRow};
  const std::array<SharedDictionaryScope, 3> scopes{
      SharedDictionaryScope::Stripe,
      SharedDictionaryScope::File,
      SharedDictionaryScope::External};

  std::mt19937 rng{kSeed};
  for (size_t iteration{0}; iteration < kIterationCount; ++iteration) {
    const auto inputType = inputTypes[std::uniform_int_distribution<size_t>{
        0, inputTypes.size() - 1}(rng)];
    const auto scope = scopes[std::uniform_int_distribution<size_t>{
        0, scopes.size() - 1}(rng)];
    const auto dictionaryTargetSet = randomDictionaryTargetSet(inputType, rng);
    const auto stripeCount = std::uniform_int_distribution<size_t>{1, 4}(rng);
    std::vector<StripeValueType> stripeValueTypes;
    stripeValueTypes.reserve(stripeCount);
    bool hasDictionaryStripe{false};
    for (size_t stripeIndex{0}; stripeIndex < stripeCount; ++stripeIndex) {
      const auto stripeValueType =
          std::uniform_int_distribution<int>{0, 1}(rng) == 0
          ? StripeValueType::Dictionary
          : StripeValueType::Direct;
      hasDictionaryStripe |= stripeValueType == StripeValueType::Dictionary;
      stripeValueTypes.push_back(stripeValueType);
    }
    if (!hasDictionaryStripe) {
      stripeValueTypes.front() = StripeValueType::Dictionary;
    }

    SCOPED_TRACE(
        fmt::format(
            "iteration={}, seed={}, inputType={}, scope={}, dictionaryTargetSet={}",
            iteration,
            kSeed,
            inputTypeName(inputType),
            SharedDictionaryScopeName::toName(scope),
            dictionaryTargetSetName(dictionaryTargetSet)));
    std::shared_ptr<const ExternalDictionaryResolver> resolver;
    if (scope == SharedDictionaryScope::External) {
      resolver = makeExternalResolver(inputType, dictionaryTargetSet);
    }
    const auto file = write(
        inputType, scope, stripeValueTypes, resolver, dictionaryTargetSet);
    auto tablet = openTablet(file, resolver);
    const auto valueStreamIds =
        sharedDictionaryValueStreamIds(*tablet, inputType);
    ASSERT_EQ(
        valueStreamIds.size(),
        expectedSharedDictionaryValueStreamCount(dictionaryTargetSet));
    for (const auto valueStreamId : valueStreamIds) {
      SCOPED_TRACE(fmt::format("valueStreamId={}", valueStreamId));
      expectDictionaryValueEncodingTypes(
          *tablet, valueStreamId, scope, stripeValueTypes);
    }
    verifyRoundTrip(file, inputType, stripeValueTypes, resolver);
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllInputTypes,
    SharedDictionaryE2EInputTypeTest,
    testing::Values(
        InputType::FlatMapScalar,
        InputType::NullableFlatMapScalar,
        InputType::FlatMapArray,
        InputType::FlatMapArrayArray,
        InputType::FlatMapMap,
        InputType::FlatMapRowValue,
        InputType::NestedScalar,
        InputType::NestedArray,
        InputType::NestedMap,
        InputType::NestedArrayRow,
        InputType::NestedMapRow),
    [](const testing::TestParamInfo<InputType>& testInfo) {
      return std::string{inputTypeName(testInfo.param)};
    });

INSTANTIATE_TEST_SUITE_P(
    InputTypes,
    SharedDictionaryE2EFlatMapDefaultValueSubfieldTest,
    testing::Values(
        InputType::FlatMapArray,
        InputType::FlatMapArrayArray,
        InputType::FlatMapMap),
    [](const testing::TestParamInfo<InputType>& testInfo) {
      return std::string{inputTypeName(testInfo.param)};
    });

INSTANTIATE_TEST_SUITE_P(
    Scopes,
    SharedDictionaryE2EFlatMapRowValueSubfieldsTest,
    testing::Values(
        SharedDictionaryScope::Stripe,
        SharedDictionaryScope::File,
        SharedDictionaryScope::External),
    [](const testing::TestParamInfo<SharedDictionaryScope>& testInfo) {
      return std::string{SharedDictionaryScopeName::toName(testInfo.param)};
    });

TEST_F(SharedDictionaryE2ETest, stripeScopeRejectsConfiguredDictionaryId) {
  auto input = makeDictionaryStripe(InputType::FlatMapScalar);
  WriterOptions options;
  addDictionary(
      options,
      InputType::FlatMapScalar,
      SharedDictionaryConfig{
          .scope = SharedDictionaryScope::Stripe, .dictionaryId = 7});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  Writer writer{
      input->type(), std::move(writeFile), *rootPool_, std::move(options)};
  NIMBLE_ASSERT_USER_THROW(
      writer.write(input),
      "Stripe shared dictionary config must leave dictionaryId unset.");
}

TEST_P(SharedDictionaryE2ETabletReaderApiTest, dictionaryApis) {
  const auto testCase = GetParam();
  std::shared_ptr<const ExternalDictionaryResolver> resolver;
  if (testCase.scope == SharedDictionaryScope::External) {
    resolver = makeExternalResolver(externalDictionaryValues());
  }
  const auto file = write(
      InputType::FlatMapScalar,
      testCase.scope,
      {StripeValueType::Dictionary, StripeValueType::Direct},
      resolver);
  auto tablet = openTablet(file, resolver);
  const auto sharedValueStreamIds =
      sharedDictionaryValueStreamIds(*tablet, InputType::FlatMapScalar);
  ASSERT_EQ(sharedValueStreamIds.size(), 1);
  const auto valueStreamId = sharedValueStreamIds.front();

  EXPECT_EQ(tablet->hasStripeDictionaries(), testCase.hasStripeDictionaries);
  EXPECT_EQ(
      tablet->hasFileOrExternalDictionaries(),
      testCase.hasFileOrExternalDictionaries);
  const auto resolvedAlphabet =
      tablet->resolveDictionaryAlphabet(valueStreamId);
  EXPECT_EQ(resolvedAlphabet != nullptr, testCase.resolvesDictionaryAlphabet);
  if (testCase.resolvesDictionaryAlphabet) {
    ASSERT_NE(resolvedAlphabet, nullptr);
    EXPECT_EQ(
        materializedAlphabetValues(*resolvedAlphabet),
        expectedResolvedAlphabetValues(file, testCase.scope));
  } else {
    EXPECT_EQ(resolvedAlphabet, nullptr);
  }

  const auto stripeDictionaryStreamId =
      tablet->stripeDictionaryStreamId(valueStreamId);
  EXPECT_EQ(
      stripeDictionaryStreamId.has_value(),
      testCase.hasStripeDictionaryStreamId);
  if (!testCase.hasStripeDictionaryStreamId) {
    const std::array valueStreamIds{valueStreamId};
    EXPECT_TRUE(tablet->stripeDictionaryStreamIds(valueStreamIds).empty());
    return;
  }

  ASSERT_TRUE(stripeDictionaryStreamId.has_value());
  const std::array valueStreamIds{
      valueStreamId, stripeDictionaryStreamId.value()};
  const auto dictionaryStreamIds =
      tablet->stripeDictionaryStreamIds(valueStreamIds);
  ASSERT_EQ(dictionaryStreamIds.size(), 1);
  EXPECT_EQ(dictionaryStreamIds.at(valueStreamId), stripeDictionaryStreamId);
  EXPECT_FALSE(dictionaryStreamIds.contains(stripeDictionaryStreamId.value()));

  const std::array dictionaryStreamIdsToLoad{stripeDictionaryStreamId.value()};
  auto dictionaryStreams =
      tablet->load(tablet->stripeIdentifier(0), dictionaryStreamIdsToLoad);
  ASSERT_EQ(dictionaryStreams.size(), 1);
  ASSERT_NE(dictionaryStreams.front(), nullptr);
  std::shared_ptr<const StreamLoader> dictionaryStreamOwner{
      std::move(dictionaryStreams.front())};
  auto alphabet = SharedDictionaryAlphabet::create(
      dictionaryStreamOwner->getStream(),
      dictionaryStreamOwner,
      leafPool_.get());
  ASSERT_NE(alphabet, nullptr);
  EXPECT_EQ(
      materializedAlphabetValues(*alphabet), kDictionaryStripeAlphabetValues);
}

INSTANTIATE_TEST_SUITE_P(
    Scopes,
    SharedDictionaryE2ETabletReaderApiTest,
    testing::Values(
        TabletReaderDictionaryApiTestCase{
            .scope = SharedDictionaryScope::Stripe,
            .hasStripeDictionaries = true,
            .hasFileOrExternalDictionaries = false,
            .hasStripeDictionaryStreamId = true,
            .resolvesDictionaryAlphabet = false},
        TabletReaderDictionaryApiTestCase{
            .scope = SharedDictionaryScope::File,
            .hasStripeDictionaries = false,
            .hasFileOrExternalDictionaries = true,
            .hasStripeDictionaryStreamId = false,
            .resolvesDictionaryAlphabet = true},
        TabletReaderDictionaryApiTestCase{
            .scope = SharedDictionaryScope::External,
            .hasStripeDictionaries = false,
            .hasFileOrExternalDictionaries = true,
            .hasStripeDictionaryStreamId = false,
            .resolvesDictionaryAlphabet = true}),
    [](const testing::TestParamInfo<TabletReaderDictionaryApiTestCase>&
           testInfo) {
      return std::string{
          SharedDictionaryScopeName::toName(testInfo.param.scope)};
    });

TEST_F(SharedDictionaryE2ETest, dictionaryConfigRejected) {
  for (const auto target :
       {InvalidDictionaryTarget::FlatMapWholeRowValue,
        InvalidDictionaryTarget::RegularRowColumnValue}) {
    SCOPED_TRACE(fmt::format("target={}", invalidDictionaryTargetName(target)));
    velox::test::VectorMaker maker{leafPool_.get()};
    velox::RowVectorPtr input;
    WriterOptions options;
    switch (target) {
      case InvalidDictionaryTarget::FlatMapWholeRowValue:
        input = maker.rowVector(
            {"features"},
            {maker.mapVector(
                std::vector<velox::vector_size_t>{0},
                std::vector<velox::vector_size_t>{1},
                maker.flatVector<int64_t>({10}),
                maker.rowVector(
                    {"a", "b"},
                    {maker.flatVector<int64_t>({1}),
                     maker.flatVector<int64_t>({2})}))});
        options.flatMapColumns.emplace("features", std::set<std::string>{});
        addFlatmapDictionary(
            options,
            sharedDictionaryConfig(
                SharedDictionaryScope::File, /*dictionaryId=*/17));
        break;
      case InvalidDictionaryTarget::RegularRowColumnValue:
        input = maker.rowVector(
            {"nested"},
            {maker.rowVector(
                {"a", "b"},
                {maker.flatVector<int32_t>({1}),
                 maker.flatVector<int32_t>({2})})});
        useSharedDictionarySelectionPolicy(options);
        options.experimentalSharedDictionaryEncoding =
            SharedDictionaryEncodingConfig::builder()
                .addColumnDictionary(
                    "nested",
                    sharedDictionaryConfig(
                        SharedDictionaryScope::File, /*dictionaryId=*/17))
                .build();
        break;
    }
    ASSERT_NE(input, nullptr);
    auto writeInvalidTarget = [&]() {
      std::string file;
      auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
      Writer writer{
          input->type(), std::move(writeFile), *rootPool_, std::move(options)};
      writer.write(input);
    };
    NIMBLE_ASSERT_USER_THROW(
        writeInvalidTarget(), "must resolve to an integer scalar");
  }
}

TEST_F(
    SharedDictionaryE2ETest,
    fileScopeSharedDictionaryStreamUsesChunkCompression) {
  constexpr size_t kStripeCount{2};
  const auto file = writeWithCompressedFileDictionary(kStripeCount);
  auto tablet = openTablet(file);
  const auto valueStreamIds =
      sharedDictionaryValueStreamIds(*tablet, InputType::FlatMapScalar);
  ASSERT_EQ(valueStreamIds.size(), 1);
  const auto valueStreamId = valueStreamIds.front();
  const auto encodingTypes =
      streamEncodingTypesByStripe(*tablet, valueStreamId);
  ASSERT_EQ(encodingTypes.size(), kStripeCount);
  expectAllStripeEncodingTypes(encodingTypes, EncodingType::SharedDictionary);
  for (uint32_t stripeIndex{0}; stripeIndex < kStripeCount; ++stripeIndex) {
    SCOPED_TRACE(fmt::format("stripeIndex={}", stripeIndex));
    EXPECT_EQ(
        dictionaryValueChunkCompressionType(
            file, InputType::FlatMapScalar, stripeIndex),
        CompressionType::Zstd);
  }
  verifyRoundTrip(
      file,
      InputType::FlatMapScalar,
      std::vector<StripeValueType>(kStripeCount, StripeValueType::Dictionary));
}

TEST_F(SharedDictionaryE2ETest, fileScopePrebuiltAlphabetPreservesEncoding) {
  constexpr size_t kStripeCount{2};
  for (const auto alphabetOrder :
       {FileDictionaryAlphabetOrder::Sorted,
        FileDictionaryAlphabetOrder::Unsorted}) {
    SCOPED_TRACE(
        fmt::format(
            "alphabetOrder={}",
            fileDictionaryAlphabetOrderName(alphabetOrder)));
    const auto file = writeWithFileDictionary(kStripeCount, alphabetOrder);
    auto tablet = openTablet(file);
    const auto valueStreamIds =
        sharedDictionaryValueStreamIds(*tablet, InputType::FlatMapScalar);
    ASSERT_EQ(valueStreamIds.size(), 1);
    const auto valueStreamId = valueStreamIds.front();
    const auto encodingTypes =
        streamEncodingTypesByStripe(*tablet, valueStreamId);
    ASSERT_EQ(encodingTypes.size(), kStripeCount);
    expectAllStripeEncodingTypes(encodingTypes, EncodingType::SharedDictionary);
    EXPECT_EQ(fileDictionaryValues(file), fileDictionaryValues(alphabetOrder));
    EXPECT_EQ(
        fileDictionaryAlphabetEncodingType(file), EncodingType::FixedBitWidth);
    verifyFileDictionaryRoundTrip(file, alphabetOrder, kStripeCount);
  }
}

TEST_P(SharedDictionaryE2EInputTypeTest, externalScopeRoundTrip) {
  const auto inputType = GetParam();
  const std::vector<StripeValueType> stripeValueTypes{
      StripeValueType::Dictionary, StripeValueType::Direct};
  auto resolver = makeExternalResolver(
      externalDictionaryValues(dictionaryValueCount(inputType)));
  const auto file = write(
      inputType, SharedDictionaryScope::External, stripeValueTypes, resolver);
  auto tablet = openTablet(file, resolver);
  const auto valueStreamIds =
      sharedDictionaryValueStreamIds(*tablet, inputType);
  ASSERT_EQ(valueStreamIds.size(), 1);
  for (const auto valueStreamId : valueStreamIds) {
    SCOPED_TRACE(fmt::format("valueStreamId={}", valueStreamId));
    expectDictionaryValueEncodingTypes(
        *tablet,
        valueStreamId,
        SharedDictionaryScope::External,
        stripeValueTypes);
  }
  verifyRoundTrip(file, inputType, stripeValueTypes, resolver);
}

TEST_F(SharedDictionaryE2ETest, externalDictionaryFailures) {
  for (const auto failure :
       {ExternalDictionaryFailure::MissingResolver,
        ExternalDictionaryFailure::MissingValue}) {
    SCOPED_TRACE(
        fmt::format("failure={}", externalDictionaryFailureName(failure)));
    switch (failure) {
      case ExternalDictionaryFailure::MissingResolver: {
        auto resolver = makeExternalResolver(externalDictionaryValues());
        const auto file = write(
            InputType::FlatMapScalar,
            SharedDictionaryScope::External,
            {StripeValueType::Dictionary, StripeValueType::Direct},
            resolver);
        auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
        VeloxReader reader{readFile, *leafPool_};
        velox::VectorPtr output;
        NIMBLE_ASSERT_USER_THROW(
            reader.next(kStripeRows, output),
            "External shared dictionary 17 requires an "
            "ExternalDictionaryResolver.");
        break;
      }
      case ExternalDictionaryFailure::MissingValue: {
        auto resolver = makeExternalResolver(kDictionaryStripeAlphabetValues);
        NIMBLE_ASSERT_USER_THROW(
            write(
                InputType::FlatMapScalar,
                SharedDictionaryScope::External,
                {StripeValueType::Dictionary, StripeValueType::Direct},
                resolver),
            "External shared dictionary 17 does not contain value 10000.");
        break;
      }
    }
  }
}

TEST_F(
    SharedDictionaryE2ETest,
    externalMissingValueAfterFirstChunkFailsStripe) {
  auto resolver = makeExternalResolver(kDictionaryStripeAlphabetValues);
  velox::test::VectorMaker maker{leafPool_.get()};
  auto input = maker.rowVector(
      {"features"},
      {maker.mapVector<int64_t, int32_t>(
          2'000,
          [](auto /* row */) { return 1; },
          [](auto /* idx */) { return 10; },
          [](auto idx) {
            if (idx < 128) {
              return kDictionaryStripeAlphabetValues.at(
                  idx % kDictionaryStripeAlphabetValues.size());
            }
            return 42;
          })});
  WriterOptions options;
  options.maxStreamChunkRawSize = 512;
  options.minStreamChunkRawSize = 1;
  useSharedDictionarySelectionPolicy(options);
  addDictionary(
      options,
      InputType::FlatMapScalar,
      SharedDictionaryConfig{
          .scope = SharedDictionaryScope::External, .dictionaryId = 17});
  options.experimentalSharedDictionaryEncoding.externalResolver = resolver;

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  Writer writer{
      input->type(), std::move(writeFile), *rootPool_, std::move(options)};
  writer.write(input);
  NIMBLE_ASSERT_USER_THROW(
      writer.close(),
      "External shared dictionary 17 does not contain value 42.");
}

} // namespace
} // namespace facebook::nimble
