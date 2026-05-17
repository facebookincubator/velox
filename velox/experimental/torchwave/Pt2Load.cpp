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

#include "velox/experimental/torchwave/Pt2Load.h"

#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <caffe2/caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container
#include <torch/csrc/export/pt2_archive_constants.h> // @manual
#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/nativert/graph/Serialization.h>

namespace torch::wave {

namespace {

constexpr std::string_view kModelsDir =
    torch::_export::archive_spec::MODELS_DIR;
constexpr std::string_view kJsonSuffix = ".json";

/// Read a PayloadConfig JSON from the archive and return the name-to-path map.
std::unordered_map<std::string, std::string> getPayloadConfig(
    caffe2::serialize::PyTorchStreamReader& reader,
    std::string_view configFormat,
    const std::string& modelName) {
  std::string configPath = fmt::format(fmt::runtime(configFormat), modelName);
  if (!reader.hasRecord(configPath)) {
    return {};
  }
  const auto& [data, size] = reader.getRecord(configPath);
  std::string serialized{reinterpret_cast<char*>(data.get()), size};
  auto configJson = nlohmann::json::parse(serialized)
                        .template get<torch::_export::PayloadConfig>();
  std::unordered_map<std::string, std::string> paths;
  for (const auto& entry : configJson.get_config()) {
    paths[entry.first] = entry.second.get_path_name();
  }
  return paths;
}

} // namespace

std::vector<std::string> getModelNames(
    caffe2::serialize::PyTorchStreamReader& reader) {
  std::vector<std::string> names;
  for (const auto& record : reader.getAllRecords()) {
    if (record.size() > kModelsDir.size() + kJsonSuffix.size() &&
        record.compare(0, kModelsDir.size(), kModelsDir) == 0 &&
        record.compare(
            record.size() - kJsonSuffix.size(),
            kJsonSuffix.size(),
            kJsonSuffix) == 0) {
      names.push_back(record.substr(
          kModelsDir.size(),
          record.size() - kModelsDir.size() - kJsonSuffix.size()));
    }
  }
  return names;
}

LoadedModel loadPt2Model(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    const std::string& modelName) {
  std::string modelFilePath = fmt::format(
      torch::_export::archive_spec::MODELS_FILENAME_FORMAT, modelName);

  const auto& [modelData, modelSize] = reader->getRecord(modelFilePath);
  std::string modelSerialized{
      reinterpret_cast<char*>(modelData.get()), modelSize};

  auto exportedProgram = nlohmann::json::parse(modelSerialized)
                             .template get<torch::_export::ExportedProgram>();

  auto graph = torch::nativert::jsonToGraph(exportedProgram.get_graph_module());

  auto tensorPaths = getPayloadConfig(
      *reader,
      torch::_export::archive_spec::WEIGHTS_CONFIG_FILENAME_FORMAT,
      modelName);
  auto constantPaths = getPayloadConfig(
      *reader,
      torch::_export::archive_spec::CONSTANTS_CONFIG_FILENAME_FORMAT,
      modelName);

  LoadedModel loaded;
  loaded.reader = std::move(reader);
  loaded.graph = std::move(graph);
  loaded.modelName = modelName;
  loaded.tensorPaths = std::move(tensorPaths);
  loaded.constantPaths = std::move(constantPaths);
  return loaded;
}

} // namespace torch::wave
