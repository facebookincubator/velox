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

#include "velox/experimental/torchwave/PackRewrite.h"

#include <algorithm>
#include <optional>
#include <unordered_set>

#include <glog/logging.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::wave {
namespace {

constexpr std::string_view kSplitTarget =
    "torch.ops.aten.split_with_sizes.default";
constexpr std::string_view kUnpackTarget = "tw.unpack_slot";

// --- Modulus analysis ---

// What every path from a value ends in: no path at all, a remainder by one
// divisor, or something else.
constexpr int64_t kNoUse = 0;
constexpr int64_t kNotReduced = -1;

int64_t combineModulus(int64_t lhs, int64_t rhs) {
  if (lhs == kNoUse) {
    return rhs;
  }
  if (rhs == kNoUse || lhs == rhs) {
    return lhs;
  }
  return kNotReduced;
}

// Ops whose output holds values of their tensor input and nothing else, so a
// remainder after them applies to the input's values.
bool passesValues(std::string_view target) {
  static const std::unordered_set<std::string_view> kTargets{
      "torch.ops.aten.view.default",
      "torch.ops.aten._unsafe_view.default",
      "torch.ops.aten.reshape.default",
      "torch.ops.aten.clone.default",
      "torch.ops.aten.contiguous.default",
      "torch.ops.aten.alias.default",
      "torch.ops.aten.slice.Tensor",
      "torch.ops.aten.narrow.default",
      "torch.ops.aten.unsqueeze.default",
      "torch.ops.aten.squeeze.default",
      "torch.ops.aten.squeeze.dim",
      "torch.ops.aten.squeeze.dims",
      "torch.ops.aten.expand.default",
      "torch.ops.aten.cat.default",
      "torch.ops.aten.concat.default",
  };
  return kTargets.count(target) != 0;
}

std::optional<int64_t> remainderDivisor(
    const nativert::Node& node,
    const nativert::Value& value) {
  if (node.target() != "torch.ops.aten.remainder.Scalar") {
    return std::nullopt;
  }
  const auto* self = node.tryGetInput("self");
  const auto* other = node.tryGetAttribute("other");
  if (self == nullptr || self->value != &value || other == nullptr ||
      !std::holds_alternative<int64_t>(other->value)) {
    return std::nullopt;
  }
  const auto divisor = std::get<int64_t>(other->value);
  return divisor > 0 ? std::optional<int64_t>(divisor) : std::nullopt;
}

int64_t modulusOf(const nativert::Value& value);

int64_t modulusOfOutputs(const nativert::Node& node) {
  int64_t result{kNoUse};
  for (const auto* output : node.outputs()) {
    result = combineModulus(result, modulusOf(*output));
  }
  return result;
}

// The divisor that every path from 'value' ends in a remainder by, kNoUse if
// nothing reads 'value', or kNotReduced.
int64_t modulusOf(const nativert::Value& value) {
  int64_t result{kNoUse};
  for (const auto* user : value.users()) {
    int64_t userModulus{kNotReduced};
    if (auto divisor = remainderDivisor(*user, value)) {
      userModulus = *divisor;
    } else if (
        user->target() == "prim.ListPack" || passesValues(user->target())) {
      userModulus = modulusOfOutputs(*user);
    }
    result = combineModulus(result, userModulus);
    if (result == kNotReduced) {
      break;
    }
  }
  return result;
}

// --- Finding packed inputs ---

int32_t userInputIndex(const nativert::Graph& graph, const nativert::Value* v) {
  const auto inputs = graph.userInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] == v) {
      return static_cast<int32_t>(i);
    }
  }
  return -1;
}

// A split_with_sizes over a packed input, with the source of each length.
struct PackedSplit {
  nativert::Node* split{nullptr};
  nativert::Node* unpack{nullptr};
  nativert::Value* input{nullptr};
  nativert::Node* sizesPack{nullptr};
  PackRecipe recipe;
};

// Why 'split' does not split a packed input, or nullopt if it does. Fills
// 'site' when it does.
std::optional<std::string>
checkSplit(nativert::Graph& graph, nativert::Node& split, PackedSplit& site) {
  const auto* self = split.tryGetInput("self");
  if (self == nullptr) {
    return "no self";
  }
  auto* input = self->value;
  const auto inputIndex = userInputIndex(graph, input);
  if (inputIndex < 0) {
    return "self is not a user input";
  }
  if (input->users().size() != 1) {
    return "self has other users";
  }
  const auto& metas = graph.tensorValuesMeta();
  auto meta = metas.find(std::string(input->name()));
  if (meta == metas.end() || meta->second.dim() != 1 ||
      meta->second.dtype() != c10::ScalarType::Long) {
    return "self is not 1-d int64";
  }
  if (const auto* dim = split.tryGetAttribute("dim"); dim != nullptr &&
      (!std::holds_alternative<int64_t>(dim->value) ||
       std::get<int64_t>(dim->value) != 0)) {
    return "split is not along dim 0";
  }
  if (split.outputs().size() != 1 || split.outputs()[0]->users().size() != 1 ||
      split.outputs()[0]->users()[0]->target() != "prim.ListUnpack") {
    return "split outputs are not unpacked";
  }
  auto* unpack = split.outputs()[0]->users()[0];

  std::vector<PackSlot> slots;
  if (const auto* sizesInput = split.tryGetInput("split_sizes")) {
    auto* pack = sizesInput->value->producer();
    if (pack == nullptr || pack->target() != "prim.ListPack") {
      return "split sizes are not a list";
    }
    std::unordered_set<const nativert::Value*> seen;
    for (const auto& element : pack->inputs()) {
      const auto lengthIndex = userInputIndex(graph, element.value);
      if (lengthIndex < 0) {
        return "a split size is not a user input";
      }
      if (!seen.insert(element.value).second) {
        return "two slots share a length";
      }
      slots.push_back({.lengthInput = lengthIndex});
    }
    site.sizesPack = pack;
  } else if (const auto* sizesAttr = split.tryGetAttribute("split_sizes");
             sizesAttr != nullptr &&
             std::holds_alternative<std::vector<int64_t>>(sizesAttr->value)) {
    for (auto length : std::get<std::vector<int64_t>>(sizesAttr->value)) {
      slots.push_back({.constLength = length});
    }
  } else {
    return "no split sizes";
  }
  if (slots.size() != unpack->outputs().size()) {
    return "split sizes and outputs differ in number";
  }
  for (size_t k = 0; k < slots.size(); ++k) {
    const auto* output = unpack->outputs()[k];
    slots[k].used = !output->users().empty();
    const auto modulus = modulusOf(*output);
    slots[k].modulus = modulus > 0 ? modulus : 0;
  }
  site.split = &split;
  site.unpack = unpack;
  site.input = input;
  site.recipe.inputName = std::string(input->name());
  site.recipe.inputIndex = inputIndex;
  site.recipe.slots = std::move(slots);
  return std::nullopt;
}

// --- Rewrite ---

nativert::Value* addUnpackNode(
    nativert::Graph& graph,
    const PackedSplit& site,
    int64_t slot,
    std::string_view outputName,
    nativert::Node* before) {
  std::vector<nativert::NamedArgument> inputs;
  const auto& packSlot = site.recipe.slots.at(slot);
  if (packSlot.lengthInput >= 0) {
    inputs.push_back({"size", site.sizesPack->inputs()[slot].value});
  }
  inputs.push_back({"packed", site.input});
  auto* node = graph.createNode(std::string(kUnpackTarget), std::move(inputs));
  if (packSlot.lengthInput < 0) {
    node->addAttribute({"size", packSlot.constLength});
  }
  node->addAttribute({"slot", slot});
  auto* output = node->addOutput(outputName, nativert::Type::Kind::Tensor);
  graph.insertBefore(node, before);
  return output;
}

// Replaces the split and its unpack with one tw.unpack_slot per read slot.
// Each new output takes the name of the split output it replaces, so the
// graph's tensor metadata, which is keyed by name, still describes it. A name
// is free only once the old value is gone, hence the two passes.
void rewriteSplit(nativert::Graph& graph, PackedSplit& site) {
  const auto numSlots = static_cast<int64_t>(site.recipe.slots.size());
  std::vector<std::string> names(numSlots);
  std::vector<nativert::Value*> temporaries(numSlots, nullptr);
  for (int64_t k = 0; k < numSlots; ++k) {
    auto* output = site.unpack->outputs()[k];
    if (output->users().empty()) {
      continue;
    }
    names[k] = std::string(output->name());
    temporaries[k] =
        addUnpackNode(graph, site, k, graph.getUniqueValueName(), site.split);
    graph.replaceAllUses(output, temporaries[k]);
  }
  graph.removeNode(site.unpack);
  graph.removeNode(site.split);
  for (int64_t k = 0; k < numSlots; ++k) {
    if (temporaries[k] == nullptr) {
      continue;
    }
    auto* temporaryNode = temporaries[k]->producer();
    auto* output = addUnpackNode(graph, site, k, names[k], temporaryNode);
    graph.replaceAllUses(temporaries[k], output);
    graph.removeNode(temporaryNode);
  }
  if (site.sizesPack != nullptr &&
      site.sizesPack->outputs()[0]->users().empty()) {
    graph.removeNode(site.sizesPack);
  }
}

std::vector<PackRewrite>& packRewrites() {
  static std::vector<PackRewrite> rewrites;
  return rewrites;
}

std::optional<int64_t> intAttribute(
    const nativert::Node& node,
    std::string_view name) {
  const auto* attribute = node.tryGetAttribute(std::string(name));
  if (attribute == nullptr ||
      !std::holds_alternative<int64_t>(attribute->value)) {
    return std::nullopt;
  }
  return std::get<int64_t>(attribute->value);
}

} // namespace

std::vector<PackRecipe> rewritePackedInputs(nativert::Graph& graph) {
  std::vector<PackedSplit> sites;
  for (auto& node : graph.nodes()) {
    if (node.target() != kSplitTarget) {
      continue;
    }
    PackedSplit site;
    if (auto reason = checkSplit(graph, node, site)) {
      VLOG(1) << "rewritePackedInputs: not packing split "
              << node.outputs()[0]->name() << ": " << *reason;
      continue;
    }
    sites.push_back(std::move(site));
  }
  std::sort(sites.begin(), sites.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.recipe.inputIndex < rhs.recipe.inputIndex;
  });
  std::vector<PackRecipe> recipes;
  for (auto& site : sites) {
    rewriteSplit(graph, site);
    recipes.push_back(std::move(site.recipe));
  }
  if (recipes.empty()) {
    return recipes;
  }
  // The compiler indexes values by id, and the user outputs are cached
  // Values, so close the id gaps and refresh the outputs, here and after
  // every further rewrite.
  graph.renumberValues();
  graph.finalize();
  for (const auto& rewrite : packRewrites()) {
    rewrite(graph, recipes);
    graph.renumberValues();
    graph.finalize();
  }
  for (const auto& recipe : recipes) {
    LOG(INFO) << "rewritePackedInputs: " << recipe.toString();
  }
  return recipes;
}

void registerPackRewrite(PackRewrite rewrite) {
  packRewrites().push_back(std::move(rewrite));
}

std::optional<SlotRef> packedSlotOf(
    const std::vector<PackRecipe>& recipes,
    const nativert::Value& value) {
  const auto* node = value.producer();
  if (node == nullptr || node->target() != kUnpackTarget) {
    return std::nullopt;
  }
  const auto* packed = node->tryGetInput("packed");
  const auto slot = intAttribute(*node, "slot");
  if (packed == nullptr || !slot.has_value()) {
    return std::nullopt;
  }
  for (size_t i = 0; i < recipes.size(); ++i) {
    if (recipes[i].inputName == packed->value->name() && *slot >= 0 &&
        *slot < static_cast<int64_t>(recipes[i].slots.size())) {
      return SlotRef{static_cast<int32_t>(i), static_cast<int32_t>(*slot)};
    }
  }
  return std::nullopt;
}

std::optional<LengthsRow> lengthsRowOf(
    const nativert::Graph& graph,
    const nativert::Value& value) {
  const auto* unpack = value.producer();
  if (unpack == nullptr || unpack->target() != "prim.ListUnpack") {
    return std::nullopt;
  }
  const auto& outputs = unpack->outputs();
  const auto row = std::find(outputs.begin(), outputs.end(), &value);
  const auto* unbind = unpack->inputs()[0].value->producer();
  if (unbind == nullptr || unbind->target() != "torch.ops.aten.unbind.int" ||
      unbind->outputs()[0]->users().size() != 1 ||
      intAttribute(*unbind, "dim").value_or(0) != 0) {
    return std::nullopt;
  }
  const auto* view = unbind->inputs()[0].value->producer();
  if (view == nullptr || view->target() != "torch.ops.aten.view.default" ||
      view->outputs()[0]->users().size() != 1) {
    return std::nullopt;
  }
  const auto* size = view->tryGetAttribute("size");
  if (size == nullptr ||
      !std::holds_alternative<std::vector<int64_t>>(size->value) ||
      std::get<std::vector<int64_t>>(size->value).size() != 2 ||
      std::get<std::vector<int64_t>>(size->value)[0] !=
          static_cast<int64_t>(outputs.size())) {
    return std::nullopt;
  }
  auto* input = view->inputs()[0].value;
  const auto inputIndex = userInputIndex(graph, input);
  if (inputIndex < 0 || input->users().size() != 1) {
    return std::nullopt;
  }
  return LengthsRow{
      inputIndex,
      static_cast<int32_t>(row - outputs.begin()),
      static_cast<int64_t>(outputs.size())};
}

std::optional<std::string> setSlotWindow(
    const nativert::Graph& graph,
    PackRecipe& recipe,
    int32_t slot,
    const LengthsRow& lengths,
    const PackSlot& window) {
  auto& packSlot = recipe.slots.at(slot);
  if (packSlot.hasWindow()) {
    return "the slot already has a window";
  }
  if (packSlot.lengthInput < 0) {
    return "the slot's length is a constant";
  }
  const auto* length = graph.userInputs()[packSlot.lengthInput];
  if (length->users().size() != 1 ||
      length->users()[0]->target() != kUnpackTarget) {
    return "the slot's length is read elsewhere";
  }
  if (recipe.lengthsInput >= 0 &&
      (recipe.lengthsInput != lengths.input ||
       recipe.lengthsRows != lengths.numRows)) {
    return "the recipe's windows take their lengths from another input";
  }
  for (const auto& other : recipe.slots) {
    if (other.hasWindow() && other.lengthsRow == lengths.row) {
      return "another slot has the same lengths row";
    }
  }
  recipe.lengthsInput = lengths.input;
  recipe.lengthsRows = lengths.numRows;
  packSlot.lengthsRow = lengths.row;
  packSlot.guardGroup = window.guardGroup;
  packSlot.maxLength = window.maxLength;
  packSlot.keepTail = window.keepTail;
  return std::nullopt;
}

} // namespace torch::wave
