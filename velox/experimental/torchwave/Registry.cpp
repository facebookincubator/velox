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

#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Utils.h"

#include <c10/util/StringUtil.h>

namespace torch::wave {

std::unordered_map<std::string, Metadata>& Registry::registry() {
  static std::unordered_map<std::string, Metadata> map;
  return map;
}

bool Metadata::isStandalone(NodeCP node, const ValueTypes& types) const {
  if (isStandalone_) {
    return true;
  }
  if (only1d) {
    for (const auto& input : node->inputs()) {
      auto* value = input.value;
      if (value->type().kind() == nativert::Type::Kind::TensorList) {
        for (auto* elem : value->getListElements()) {
          auto r = types.rank(elem);
          if (r < 0 || r > 1) {
            return true;
          }
        }
      } else if (value->type().kind() == nativert::Type::Kind::Tensor) {
        auto r = types.rank(value);
        if (r < 0 || r > 1) {
          return true;
        }
      }
    }
  }
  if (isStandaloneFunc) {
    return isStandaloneFunc(node, types);
  }
  return false;
}

void Metadata::defaultInputMeta() {
  TORCH_CHECK(functionSchema, "defaultInputMeta requires functionSchema");
  if (argumentMeta.empty()) {
    argumentMeta.resize(functionSchema->arguments().size());
  }
}

void Metadata::defaultOutputMeta() {
  TORCH_CHECK(functionSchema, "defaultOutputMeta requires functionSchema");
  if (returnMeta.empty()) {
    returnMeta.resize(functionSchema->returns().size());
  }
}

void Registry::registerMetadata(std::string_view op, Metadata metadata) {
  if (metadata.functionSchema) {
    auto numArgs = metadata.functionSchema->arguments().size();
    auto numReturns = metadata.functionSchema->returns().size();
    TORCH_CHECK(
        metadata.argumentMeta.size() == numArgs,
        op,
        ": argumentMeta size ",
        metadata.argumentMeta.size(),
        " != schema argument count ",
        numArgs);
    TORCH_CHECK(
        metadata.returnMeta.size() == numReturns,
        op,
        ": returnMeta size ",
        metadata.returnMeta.size(),
        " != schema return count ",
        numReturns);
    for (const auto& ret : metadata.returnMeta) {
      for (auto idx : ret.sizeArgs.ordinal) {
        TORCH_CHECK(
            idx >= 0 && idx < static_cast<int32_t>(numArgs),
            op,
            ": sizeArgs index ",
            idx,
            " out of range [0, ",
            numArgs,
            ")");
      }
    }
    for (auto idx : metadata.typeTemplateParams) {
      TORCH_CHECK(
          idx >= 0 && idx < static_cast<int32_t>(numArgs),
          op,
          ": typeTemplateParams index ",
          idx,
          " out of range [0, ",
          numArgs,
          ")");
    }
    if (metadata.inputFromPreviousKernel.has_value()) {
      auto idx = metadata.inputFromPreviousKernel.value();
      TORCH_CHECK(
          idx >= 0 && idx < static_cast<int32_t>(numArgs),
          op,
          ": inputFromPreviousKernel index ",
          idx,
          " out of range [0, ",
          numArgs,
          ")");
    }
  }
  registry()[std::string(op)] = std::move(metadata);
}

// NOLINTNEXTLINE(facebook-hte-NullableReturn)
const Metadata* Registry::metadata(std::string_view op) {
  auto& map = registry();
  auto it = map.find(std::string(op));
  if (it == map.end()) {
    return nullptr;
  }
  return &it->second;
}

Metadata Registry::unregister(std::string_view name) {
  auto& map = registry();
  auto it = map.find(std::string(name));
  TORCH_CHECK(it != map.end(), "Registry entry not found: ", name);
  auto metadata = std::move(it->second);
  map.erase(it);
  return metadata;
}

void Registry::restoreRegistry(std::string_view name, Metadata metadata) {
  registry()[std::string(name)] = std::move(metadata);
}

void Registry::registerElementwise(std::string_view qualifiedName) {
  auto atoms = c10::split(qualifiedName, '.');
  TORCH_CHECK(atoms.size() >= 3, "Invalid qualified op name: ", qualifiedName);
  auto opName = atoms[atoms.size() - 2];

  const auto* schema = findFunctionSchema(qualifiedName);
  TORCH_CHECK(schema, "FunctionSchema not found for: ", qualifiedName);

  Metadata md;

  md.functionSchema = schema;
  md.inPlaceIfLastUse = true;
  md.argumentMeta.resize(
      schema->arguments().size(), ArgumentMeta{.isRegister = true});
  md.returnMeta = {ArgumentMeta{.isRegister = true, .sizeArgs = {{0}, {}}}};
  md.elementwise = std::make_unique<ElementwiseOp>();
  md.elementwise->functionName = fmt::format("__{}", opName);

  registerMetadata(qualifiedName, std::move(md));
}

void Registry::registerElementwiseOp(
    std::string_view qualifiedName,
    std::string_view elementwiseFuncName,
    bool isStandalone) {
  const auto* schema = findFunctionSchema(qualifiedName);
  TORCH_CHECK(schema, "FunctionSchema not found for: ", qualifiedName);

  Metadata md;

  md.functionSchema = schema;
  md.inPlaceIfLastUse = true;
  md.isStandalone_ = isStandalone;
  md.argumentMeta.resize(
      schema->arguments().size(), ArgumentMeta{.isRegister = true});
  md.returnMeta = {ArgumentMeta{.isRegister = true, .sizeArgs = {{0}, {}}}};
  md.elementwise = std::make_unique<ElementwiseOp>();
  md.elementwise->functionName = std::string(elementwiseFuncName);

  registerMetadata(qualifiedName, std::move(md));
}

std::vector<std::unique_ptr<c10::FunctionSchema>>& Registry::schemaStorage() {
  static std::vector<std::unique_ptr<c10::FunctionSchema>> storage;
  return storage;
}

const c10::FunctionSchema* Registry::ownSchema(
    std::unique_ptr<c10::FunctionSchema> schema) {
  auto* ptr = schema.get();
  schemaStorage().push_back(std::move(schema));
  return ptr;
}

MetadataBuilder::MetadataBuilder(std::string_view qualifiedName)
    : name_(qualifiedName) {
  md_.functionSchema = findFunctionSchema(qualifiedName);
  TORCH_CHECK(
      md_.functionSchema, "FunctionSchema not found for: ", qualifiedName);
}

MetadataBuilder::MetadataBuilder(std::unique_ptr<c10::FunctionSchema> schema) {
  name_ = schema->name();
  if (!schema->overload_name().empty()) {
    name_ += "." + schema->overload_name();
  }
  md_.functionSchema = Registry::ownSchema(std::move(schema));
}

MetadataBuilder& MetadataBuilder::sizeShortcut(SizeShortcut shortcut) {
  builderSizeShortcut_ = shortcut;
  sizeShortcutSet_ = true;
  return *this;
}

MetadataBuilder& MetadataBuilder::sizeOrdinal(std::vector<int32_t> ordinal) {
  builderSizeArgs_.ordinal = std::move(ordinal);
  sizeArgsSet_ = true;
  return *this;
}

MetadataBuilder& MetadataBuilder::sizeArgsList(std::vector<bool> isList) {
  builderSizeArgs_.isList = std::move(isList);
  sizeArgsSet_ = true;
  return *this;
}

MetadataBuilder& MetadataBuilder::argumentMeta(std::vector<ArgumentMeta> meta) {
  md_.argumentMeta = std::move(meta);
  return *this;
}

MetadataBuilder& MetadataBuilder::defaultInputMeta() {
  md_.defaultInputMeta();
  return *this;
}

MetadataBuilder& MetadataBuilder::returnMeta(std::vector<ArgumentMeta> meta) {
  md_.returnMeta = std::move(meta);
  return *this;
}

MetadataBuilder& MetadataBuilder::defaultOutputMeta() {
  md_.defaultOutputMeta();
  return *this;
}

MetadataBuilder& MetadataBuilder::hasBarrier(bool val) {
  md_.hasBarrier = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::singleBlockIfFused(bool val) {
  md_.singleBlockIfFused = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::inputFromPreviousKernel(int32_t ordinal) {
  md_.inputFromPreviousKernel = ordinal;
  return *this;
}

MetadataBuilder& MetadataBuilder::multiBlockReturnBarrier(bool val) {
  md_.multiBlockReturnBarrier = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::alwaysSingleBlock(bool val) {
  md_.alwaysSingleBlock = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::metadataGetter(bool val) {
  md_.isMetadataGetter = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::makeMultiKernelVariant(
    std::function<nativert::Node*(NodeCP, WaveGraph*)> func) {
  md_.makeMultiKernelVariant = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::cgVariant(
    std::function<nativert::Node*(NodeCP, WaveGraph*)> func) {
  md_.cgVariant = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::numBarriers(int32_t val) {
  md_.numBarriers = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::arithmeticPromotion(bool val) {
  md_.arithmeticPromotion = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::inPlaceIfLastUse(bool val) {
  md_.inPlaceIfLastUse = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::isStandalone(bool val) {
  md_.isStandalone_ = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::only1d(bool val) {
  md_.only1d = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::isStandaloneFunc(
    std::function<bool(NodeCP, const ValueTypes&)> func) {
  md_.isStandaloneFunc = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::cost(float val) {
  md_.cost = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::costFunction(
    std::function<float(NodeCP, const Metadata&)> func) {
  md_.costFunction = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::viewOfArg(int32_t ordinal) {
  md_.viewOfArg = ordinal;
  return *this;
}

MetadataBuilder& MetadataBuilder::shapeAttr(std::string name) {
  md_.shapeAttr = std::move(name);
  return *this;
}

MetadataBuilder& MetadataBuilder::ignoreAttrs(std::vector<std::string> attrs) {
  md_.ignoreAttrs = std::move(attrs);
  return *this;
}

MetadataBuilder& MetadataBuilder::rankArgument(int32_t ordinal) {
  md_.rankArgument = ordinal;
  return *this;
}

MetadataBuilder& MetadataBuilder::outputConstraints(
    std::function<std::vector<ValueConstraint>(NodeCP, const ValueTypes&)>
        func) {
  md_.outputConstraints = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::maybeReplace(
    std::function<std::vector<
        std::pair<ValueCP, ValueCP>>(NodeCP, ValueTypes&, WaveGraph&)> func) {
  md_.maybeReplace = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::normalize(
    std::function<void(nativert::Node*, const ValueTypes&)> func) {
  md_.normalize = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::generateCall(
    std::function<void(std::stringstream&, NodeCP, std::vector<std::string>)>
        func) {
  md_.generateCall = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::specialForm(
    std::function<void(NodeCP, const std::vector<ResultSpec>&, CompileCtx*)>
        func) {
  md_.specialForm = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::headerFile(std::string file) {
  md_.headerFile = std::move(file);
  return *this;
}

MetadataBuilder& MetadataBuilder::deviceFunc(std::string func) {
  md_.deviceFunc = std::move(func);
  return *this;
}

MetadataBuilder& MetadataBuilder::sharedDecls(
    std::vector<std::pair<std::string, std::string>> decls) {
  md_.sharedDecls = std::move(decls);
  return *this;
}

MetadataBuilder& MetadataBuilder::dynamicSharedDecls(
    std::vector<std::pair<int32_t, std::string>> decls) {
  md_.dynamicSharedDecls = std::move(decls);
  return *this;
}

MetadataBuilder& MetadataBuilder::typeTemplateParams(
    std::vector<int32_t> params) {
  md_.typeTemplateParams = std::move(params);
  return *this;
}

MetadataBuilder& MetadataBuilder::hasBlockSizeTemplateParam(bool val) {
  md_.hasBlockSizeTemplateParam = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::hasDtypeTemplateParam(bool val) {
  md_.hasDtypeTemplateParam = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::templateAttrs(
    std::vector<std::string> attrs) {
  md_.templateAttrs = std::move(attrs);
  return *this;
}

MetadataBuilder& MetadataBuilder::setOutputs(
    std::function<void(
        KernelOperation* op,
        NodeCP node,
        const std::unordered_set<ValueCP>& subgraphInputs,
        std::vector<ValueCP>& outputValues,
        std::vector<OutputDesc>& outputDescs,
        bool inMemory,
        bool callerIsElementwise)> func) {
  md_.setOutputs = std::move(func);
  return *this;
}

ElementwiseOp& MetadataBuilder::ensureElementwise() {
  if (!md_.elementwise) {
    md_.elementwise = std::make_unique<ElementwiseOp>();
  }
  return *md_.elementwise;
}

MetadataBuilder& MetadataBuilder::elementwise() {
  auto atoms = c10::split(name_, '.');
  TORCH_CHECK(atoms.size() >= 3, "Invalid qualified op name: ", name_);
  auto opName = atoms[atoms.size() - 2];
  ensureElementwise().functionName = fmt::format("__{}", opName);
  builderSizeArgs_.ordinal = {0};
  sizeArgsSet_ = true;
  md_.inPlaceIfLastUse = true;
  return *this;
}

MetadataBuilder& MetadataBuilder::elementwiseFunc(std::string funcName) {
  ensureElementwise().functionName = std::move(funcName);
  builderSizeArgs_.ordinal = {0};
  sizeArgsSet_ = true;
  md_.inPlaceIfLastUse = true;
  return *this;
}

MetadataBuilder& MetadataBuilder::numArgs(int32_t n) {
  ensureElementwise().numArgs = n;
  return *this;
}

MetadataBuilder& MetadataBuilder::hasIdxArg(bool val) {
  ensureElementwise().hasIdxArg = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::hasSizeArg(bool val) {
  ensureElementwise().hasSizeArg = val;
  return *this;
}

MetadataBuilder& MetadataBuilder::hasBlockInfo(bool val) {
  ensureElementwise().hasBlockInfo = val;
  return *this;
}

Metadata MetadataBuilder::build() {
  if (md_.elementwise && md_.elementwise->numArgs == -1 && md_.functionSchema) {
    md_.elementwise->numArgs =
        static_cast<int32_t>(md_.functionSchema->arguments().size());
  }
  if (md_.elementwise && md_.elementwise->numArgs == 0) {
    builderSizeShortcut_ = SizeShortcut::kNone;
    builderSizeArgs_.ordinal.clear();
  }
  if (!md_.shapeAttr.empty() && md_.functionSchema) {
    const auto& args = md_.functionSchema->arguments();
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].name() == md_.shapeAttr) {
        auto ordinal = static_cast<int32_t>(i);
        if (std::find(
                builderSizeArgs_.ordinal.begin(),
                builderSizeArgs_.ordinal.end(),
                ordinal) == builderSizeArgs_.ordinal.end()) {
          builderSizeArgs_.ordinal.push_back(ordinal);
          sizeArgsSet_ = true;
        }
        break;
      }
    }
  }
  if (!builderSizeArgs_.ordinal.empty() &&
      builderSizeShortcut_ == SizeShortcut::kNone) {
    builderSizeShortcut_ = SizeShortcut::kMax;
  }
  if (md_.argumentMeta.empty()) {
    if (md_.elementwise) {
      md_.argumentMeta.resize(
          md_.functionSchema->arguments().size(),
          ArgumentMeta{.isRegister = true});
    } else {
      md_.defaultInputMeta();
    }
  }
  if (md_.returnMeta.empty()) {
    if (md_.elementwise) {
      md_.returnMeta = {ArgumentMeta{.isRegister = true}};
    } else {
      md_.defaultOutputMeta();
    }
  }
  if (sizeArgsSet_ || sizeShortcutSet_) {
    TORCH_CHECK(
        !md_.returnMeta.empty(),
        name_,
        ": sizeShortcut/sizeArgs set but no returnMeta");
    auto& ret = md_.returnMeta[0];
    TORCH_CHECK(
        ret.sizeShortcut == SizeShortcut::kNone && ret.sizeArgs.ordinal.empty(),
        name_,
        ": both builder-level and returnMeta[0] specify sizeShortcut/sizeArgs");
    ret.sizeShortcut = builderSizeShortcut_;
    ret.sizeArgs = std::move(builderSizeArgs_);
  }
  return std::move(md_);
}

void MetadataBuilder::registerOp() {
  Registry::registerMetadata(name_, build());
}

} // namespace torch::wave
