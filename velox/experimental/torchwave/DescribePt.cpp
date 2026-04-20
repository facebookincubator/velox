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

#include "velox/experimental/torchwave/DescribePt.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/class_type.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch::wave {

namespace {

std::unordered_map<std::string, std::vector<std::string>>&
namedTupleRegistry() {
  static std::unordered_map<std::string, std::vector<std::string>> registry;
  return registry;
}

// Recursively walks an IValue tree, printing each tensor found.
void walkIValue(
    const c10::IValue& value,
    int& position,
    const std::string& path) {
  if (value.isTensor()) {
    const auto& tensor = value.toTensor();
    std::cout << "  " << position << ": ";
    if (!path.empty()) {
      std::cout << path << "  ";
    }
    std::cout << "shape=[";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << tensor.size(i);
    }
    std::cout << "]  dtype=" << c10::toString(tensor.scalar_type()) << "\n";
    ++position;
  } else if (value.isList()) {
    const auto& list = value.toListRef();
    for (size_t i = 0; i < list.size(); ++i) {
      walkIValue(list[i], position, path + "[" + std::to_string(i) + "]");
    }
  } else if (value.isGenericDict()) {
    for (const auto& entry : value.toGenericDict()) {
      std::string key;
      if (entry.key().isString()) {
        key = std::string(entry.key().toStringRef());
      } else if (entry.key().isInt()) {
        key = std::to_string(entry.key().toInt());
      } else {
        key = "?";
      }
      walkIValue(
          entry.value(), position, path.empty() ? key : path + "." + key);
    }
  } else if (value.isTuple()) {
    const auto& elements = value.toTupleRef().elements();
    for (size_t i = 0; i < elements.size(); ++i) {
      walkIValue(elements[i], position, path + "[" + std::to_string(i) + "]");
    }
  } else if (value.isObject()) {
    const auto& obj = value.toObjectRef();
    auto cls = obj.type();
    auto numAttrs = cls ? cls->numAttributes() : 0;
    for (size_t i = 0; i < numAttrs; ++i) {
      walkIValue(
          obj.getSlot(i),
          position,
          path.empty() ? cls->getAttributeName(i)
                       : path + "." + cls->getAttributeName(i));
    }
  }
}

} // namespace

void registerNamedTuple(
    const std::string& qualifiedName,
    std::vector<std::string> fieldNames) {
  namedTupleRegistry()[qualifiedName] = std::move(fieldNames);
}

void describePt(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    std::cerr << "Cannot open " << path << "\n";
    return;
  }
  auto fileSize = file.tellg();
  file.seekg(0);
  std::vector<char> data(fileSize);
  file.read(data.data(), fileSize);

  auto& registry = namedTupleRegistry();
  if (registry.empty()) {
    auto ivalue = torch::jit::pickle_load(data);
    std::cout << "Tensors in " << path << ":\n";
    int position = 0;
    walkIValue(ivalue, position, "");
    std::cout << "Total: " << position << " tensors\n";
    return;
  }

  caffe2::serialize::PyTorchStreamReader reader(
      std::make_unique<torch::jit::VectorReader>(std::move(data)));

  torch::jit::TypeResolver typeResolver =
      [&registry](const c10::QualifiedName& qualifiedName)
      -> c10::StrongTypePtr {
    auto it = registry.find(qualifiedName.qualifiedName());
    if (it != registry.end()) {
      auto cls = c10::ClassType::create(qualifiedName, {});
      for (const auto& fieldName : it->second) {
        cls->addAttribute(fieldName, c10::AnyType::get());
      }
      return c10::StrongTypePtr(nullptr, std::move(cls));
    }
    TORCH_CHECK(false, "Unknown type: ", qualifiedName.qualifiedName());
  };

  torch::jit::ObjLoader objLoader =
      [](const at::StrongTypePtr& type, c10::IValue input)
      -> c10::intrusive_ptr<c10::ivalue::Object> {
    auto numAttrs =
        type.type_->expectRef<c10::ClassType>().numAttributes();
    auto obj = c10::ivalue::Object::create(type, numAttrs);
    if (input.isTuple()) {
      const auto& elements = input.toTupleRef().elements();
      for (size_t i = 0; i < std::min(numAttrs, elements.size()); ++i) {
        obj->setSlot(i, elements[i]);
      }
    }
    return obj;
  };

  auto ivalue = torch::jit::readArchiveAndTensors(
      "data", "", "", typeResolver, objLoader, std::nullopt, reader);
  std::cout << "Tensors in " << path << ":\n";
  int position = 0;
  walkIValue(ivalue, position, "");
  std::cout << "Total: " << position << " tensors\n";
}

} // namespace torch::wave
