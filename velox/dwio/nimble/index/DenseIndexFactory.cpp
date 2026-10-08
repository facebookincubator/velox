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
#include "velox/dwio/nimble/index/DenseIndexFactory.h"

#include <fmt/ranges.h>

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/HashIndex.h"
#include "velox/dwio/nimble/index/HashIndexWriter.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/index/IndexFactoryRegistry.h"
#include "velox/dwio/nimble/index/SortedIndex.h"
#include "velox/dwio/nimble/index/SortedIndexWriter.h"
#include "velox/dwio/nimble/tablet/HashIndexGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataCache.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"
#include "velox/dwio/nimble/tablet/SortedIndexGenerated.h"

namespace facebook::nimble::index {
namespace {

IndexFactoryRegistry<DenseIndexFactory>& factoryRegistry() {
  static auto* registry =
      new IndexFactoryRegistry<DenseIndexFactory>{IndexFamily::Dense};
  return *registry;
}

// Reads one built-in dense index directory and lazily materializes its indices
// behind the DenseIndexReader interface. 'Index' is the concrete index type
// (HashIndex or SortedIndex), 'FbDirectory' its FlatBuffer directory, and
// 'ValueInput' the input stream 'Index::create' reads its payload from.
template <typename Index, typename FbDirectory, typename ValueInput>
class BuiltInDenseIndexReader final : public DenseIndexReader {
 public:
  static std::unique_ptr<DenseIndexReader> create(
      Section rootSection,
      std::string_view indexTypeName,
      std::shared_ptr<MetadataInput> metadataInput,
      std::shared_ptr<ValueInput> valueInput,
      velox::memory::MemoryPool* pool) {
    auto reader =
        std::unique_ptr<BuiltInDenseIndexReader>(new BuiltInDenseIndexReader());
    reader->descriptors_ = parseDescriptors(rootSection, indexTypeName);

    // Always pin: findIndex returns raw pointers, so entries must stay alive.
    reader->cache_ = std::make_unique<MetadataCache<uint32_t, Index>>(
        [self = reader.get(),
         metadataInput = std::move(metadataInput),
         valueInput = std::move(valueInput),
         pool](uint32_t index) -> std::shared_ptr<Index> {
          const auto& descriptor = self->descriptors_[index];
          auto results = metadataInput->load({&descriptor.section, 1});
          NIMBLE_CHECK_EQ(results.size(), 1);
          auto metadata =
              std::make_unique<MetadataBuffer>(std::move(*results.front()));
          return Index::create(
              descriptor.columns, std::move(metadata), valueInput, pool);
        },
        /*pinEntries=*/true);
    return reader;
  }

  const IndexLookup* findIndex(
      const std::vector<std::string>& queryColumns) const override {
    for (uint32_t i = 0; i < descriptors_.size(); ++i) {
      if (descriptors_[i].columns == queryColumns) {
        return cache_->getOrCreate(i).get();
      }
    }
    return nullptr;
  }

  std::vector<std::vector<std::string>> indexColumns() const override {
    std::vector<std::vector<std::string>> columns;
    columns.reserve(descriptors_.size());
    for (const auto& descriptor : descriptors_) {
      columns.emplace_back(descriptor.columns);
    }
    return columns;
  }

 private:
  struct IndexDescriptor {
    std::vector<std::string> columns;
    MetadataSection section;
  };

  static std::vector<IndexDescriptor> parseDescriptors(
      const Section& directorySection,
      std::string_view indexTypeName) {
    const auto* root =
        flatbuffers::GetRoot<FbDirectory>(directorySection.content().data());
    NIMBLE_CHECK_NOT_NULL(root);
    const auto* indices = root->indices();
    NIMBLE_CHECK_NOT_NULL(indices);

    std::vector<IndexDescriptor> descriptors;
    descriptors.reserve(indices->size());
    for (const auto* indexSection : *indices) {
      NIMBLE_CHECK_NOT_NULL(indexSection);
      const auto* columns = indexSection->index_columns();
      NIMBLE_CHECK_NOT_NULL(columns);
      NIMBLE_CHECK_GT(
          columns->size(), 0u, "{} must have columns", indexTypeName);

      std::vector<std::string> columnNames;
      columnNames.reserve(columns->size());
      for (const auto* column : *columns) {
        NIMBLE_CHECK_NOT_NULL(column);
        columnNames.emplace_back(column->string_view());
      }
      for (const auto& descriptor : descriptors) {
        NIMBLE_CHECK(
            descriptor.columns != columnNames,
            "Duplicate {} columns: [{}]",
            indexTypeName,
            fmt::join(columnNames, ", "));
      }

      const auto* section = indexSection->section();
      NIMBLE_CHECK_NOT_NULL(section);
      descriptors.emplace_back(
          IndexDescriptor{
              std::move(columnNames),
              MetadataSection{
                  section->offset(),
                  section->size(),
                  static_cast<CompressionType>(section->compression_type())}});
    }
    return descriptors;
  }

  std::vector<IndexDescriptor> descriptors_;
  mutable std::unique_ptr<MetadataCache<uint32_t, Index>> cache_;
};

class HashIndexFactory final : public DenseIndexFactory {
 public:
  std::string_view name() const override {
    return kDenseHashIndexName;
  }

  std::unique_ptr<IndexWriter> createWriter(
      std::span<const IndexConfig*> configs,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool) const override {
    for (const auto* config : configs) {
      NIMBLE_CHECK_NOT_NULL(config);
      NIMBLE_USER_CHECK_EQ(config->family, IndexFamily::Dense);
      NIMBLE_USER_CHECK_EQ(config->name, name());
    }
    return HashIndexWriter::create(configs, inputType, pool);
  }

  std::unique_ptr<DenseIndexReader> createReader(
      Section rootSection,
      const IndexLookup::Options& options,
      velox::memory::MemoryPool* pool) const override {
    options.validate();
    auto metadataInput = createIndexMetadataInput(options);
    auto valueInput = metadataInput;
    return BuiltInDenseIndexReader<
        HashIndex,
        serialization::HashIndexDirectory,
        MetadataInput>::
        create(
            std::move(rootSection),
            "Hash index",
            std::move(metadataInput),
            std::move(valueInput),
            pool);
  }
};

class SortedIndexFactory final : public DenseIndexFactory {
 public:
  std::string_view name() const override {
    return kDenseSortedIndexName;
  }

  std::unique_ptr<IndexWriter> createWriter(
      std::span<const IndexConfig*> configs,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool) const override {
    for (const auto* config : configs) {
      NIMBLE_CHECK_NOT_NULL(config);
      NIMBLE_USER_CHECK_EQ(config->family, IndexFamily::Dense);
      NIMBLE_USER_CHECK_EQ(config->name, name());
    }
    return SortedIndexWriter::create(configs, inputType, pool);
  }

  std::unique_ptr<DenseIndexReader> createReader(
      Section rootSection,
      const IndexLookup::Options& options,
      velox::memory::MemoryPool* pool) const override {
    options.validate();
    auto metadataInput = createIndexMetadataInput(options);
    auto dataInput = createIndexDataInput(options);
    return BuiltInDenseIndexReader<
        SortedIndex,
        serialization::SortedIndexDirectory,
        velox::dwio::common::BufferedInput>::
        create(
            std::move(rootSection),
            "Sorted index",
            std::move(metadataInput),
            std::move(dataInput),
            pool);
  }
};

void ensureBuiltInFactoriesRegistered() {
  static const bool registered = [] {
    factoryRegistry().registerFactory(
        std::make_shared<const HashIndexFactory>());
    factoryRegistry().registerFactory(
        std::make_shared<const SortedIndexFactory>());
    return true;
  }();
  (void)registered;
}

} // namespace

void registerDenseIndexFactory(
    std::shared_ptr<const DenseIndexFactory> factory) {
  ensureBuiltInFactoriesRegistered();
  factoryRegistry().registerFactory(std::move(factory));
}

const DenseIndexFactory* denseIndexFactory(std::string_view name) {
  ensureBuiltInFactoriesRegistered();
  return factoryRegistry().tryGet(name);
}

} // namespace facebook::nimble::index
