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

#include "velox/exec/TaskTraceWriter.h"

#include <folly/String.h>
#include <folly/container/F14Set.h>
#include <folly/json.h>

#include "velox/common/file/File.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/trace/Trace.h"
#include "velox/exec/trace/TraceUtil.h"

namespace facebook::velox::exec::trace {
namespace {

// Returns the value to record for a config entry, replacing credentials with a
// placeholder. The trace file is written once and kept, so a secret copied
// here outlives the query that supplied it.
std::string redactIfCredential(
    const std::unordered_set<std::string>& credentialKeys,
    const std::string& key,
    std::string_view value) {
  if (credentialKeys.contains(key)) {
    return std::string(kRedactedConfigValue);
  }
  return std::string(value);
}

// Connector config property listing, comma-separated, the session-property
// keys whose values are delegated credentials. Spelled out rather than
// included from
// velox/connectors/hive/iceberg/IcebergSessionCredentials.h, which defines it
// and is the place to keep in sync: velox/exec must not depend on a connector,
// and in the CMake build that header's library is the whole Iceberg connector.
// Reading a config contract by name needs no link edge.
constexpr std::string_view kSessionCredentialKeysConfig =
    "hive.session-credential-keys";

// Returns the session property names that 'connectorId' treats as delegated
// credentials. These do not travel with the query the way the config-entry
// credentials do: a connector's credential-bearing property names come from
// its own build-time config, so the connector is the only place that knows
// them. Returns empty for an unregistered connector, which is not an error --
// session properties may name one.
folly::F14FastSet<std::string> delegatedCredentialKeys(
    const core::QueryCtx& queryCtx,
    const std::string& connectorId) {
  const auto connector =
      connector::ConnectorRegistry::tryGet(queryCtx, connectorId);
  if (connector == nullptr) {
    return {};
  }
  const auto keysCsv = connector->connectorConfig()->get<std::string>(
      std::string(kSessionCredentialKeysConfig));
  if (!keysCsv.has_value()) {
    return {};
  }

  std::vector<std::string_view> listedKeys;
  folly::split(',', keysCsv.value(), listedKeys);
  folly::F14FastSet<std::string> credentialKeys;
  for (const auto listedKey : listedKeys) {
    const auto key = folly::trimWhitespace(listedKey);
    if (!key.empty()) {
      credentialKeys.emplace(key);
    }
  }
  return credentialKeys;
}

// Returns the writable string node holding a serialized VARCHAR constant's
// value, or nullptr for anything else. ConstantTypedExpr::serialize() nests
// the literal one level down, under a Variant carrying its own type tag.
folly::dynamic* constantVarcharValue(folly::dynamic& expr) {
  if (!expr.isObject()) {
    return nullptr;
  }
  auto* variant = expr.get_ptr("value");
  if (variant == nullptr || !variant->isObject()) {
    return nullptr;
  }
  const auto* kind = variant->get_ptr("type");
  if (kind == nullptr || !kind->isString() || kind->asString() != "VARCHAR") {
    return nullptr;
  }
  auto* literal = variant->get_ptr("value");
  return (literal != nullptr && literal->isString()) ? literal : nullptr;
}

// Returns true if 'fragment' is the tail of a JSON object up to and including
// a credential field's opening quote, as in `{"metagen_key":"`. Whatever the
// expression concatenates next is that field's value.
bool opensCredentialField(
    const std::unordered_set<std::string>& credentialConfigKeys,
    std::string_view fragment) {
  for (const auto& key : credentialConfigKeys) {
    const auto opener = fmt::format("\"{}\":\"", key);
    if (fragment.size() >= opener.size() &&
        fragment.compare(
            fragment.size() - opener.size(), opener.size(), opener) == 0) {
      return true;
    }
  }
  return false;
}

// Replaces every VARCHAR constant in 'expr' and below with the placeholder.
void redactConstantsIn(folly::dynamic& expr) {
  if (auto* literal = constantVarcharValue(expr)) {
    *literal = std::string(kRedactedConfigValue);
  }
  if (expr.isObject()) {
    for (const auto& key : expr.keys()) {
      redactConstantsIn(expr[key]);
    }
  } else if (expr.isArray()) {
    for (auto& element : expr) {
      redactConstantsIn(element);
    }
  }
}

// Replaces credential values carried inside a serialized plan with the
// placeholder. Query configs are not the only way a key reaches the trace
// file: an AI function's options object is built in the plan, and
// ConstantTypedExpr::serialize() writes its constants verbatim, so a key
// supplied in SQL lands in the plan block. Rewrites only the trace file's copy
// -- ConstantTypedExpr::serialize() itself must stay faithful, since the same
// serialization carries real plans between workers.
//
// Two shapes reach here. The options object may arrive already assembled, as
// one constant holding whole JSON. It may instead arrive as a concatenation,
// which is what the Presto rewriter emits: the field opener, the key, and the
// closing quote are separate arguments, so the secret is a bare constant with
// nothing in it to recognize. The fragment in front of it is what identifies
// it, which is why the argument list is walked pairwise rather than leaf by
// leaf.
//
// The plan block is deserialized back into a core::PlanNode on replay, so the
// placeholder has to leave the value a valid JSON string, which it does.
void redactPlanCredentials(
    const std::unordered_set<std::string>& credentialConfigKeys,
    folly::dynamic& planObj) {
  if (planObj.isObject()) {
    auto* inputs = planObj.get_ptr("inputs");
    if (inputs != nullptr && inputs->isArray()) {
      for (size_t i = 0; i + 1 < inputs->size(); ++i) {
        auto* fragment = constantVarcharValue((*inputs)[i]);
        if (fragment != nullptr &&
            opensCredentialField(credentialConfigKeys, fragment->asString())) {
          redactConstantsIn((*inputs)[i + 1]);
        }
      }
    }
    for (const auto& key : planObj.keys()) {
      redactPlanCredentials(credentialConfigKeys, planObj[key]);
    }
    return;
  }
  if (planObj.isArray()) {
    for (auto& element : planObj) {
      redactPlanCredentials(credentialConfigKeys, element);
    }
    return;
  }
  if (!planObj.isString()) {
    return;
  }

  const auto value = planObj.asString();
  // Only an options object can carry a key. Parsing every string constant in
  // a plan would be wasteful, and most of them are not JSON at all.
  if (value.find('{') == std::string::npos) {
    return;
  }
  folly::dynamic options;
  try {
    options = folly::parseJson(value);
  } catch (const std::exception&) {
    return;
  }
  if (!options.isObject()) {
    return;
  }

  std::vector<std::string> credentialKeys;
  for (const auto& [key, optionValue] : options.items()) {
    if (key.isString() && optionValue.isString() &&
        credentialConfigKeys.contains(key.asString())) {
      credentialKeys.push_back(key.asString());
    }
  }
  // Leave the constant byte-for-byte alone when it holds no credential, so a
  // plan that never touched an LLM traces exactly as it did before.
  if (credentialKeys.empty()) {
    return;
  }
  for (const auto& key : credentialKeys) {
    options[key] = std::string(kRedactedConfigValue);
  }
  planObj = folly::toJson(options);
}

} // namespace

TaskTraceMetadataWriter::TaskTraceMetadataWriter(
    std::string traceDir,
    std::string traceNodeId,
    memory::MemoryPool* /* pool */)
    : traceDir_(std::move(traceDir)),
      traceNodeId_(std::move(traceNodeId)),
      fs_(filesystems::getFileSystem(traceDir_, nullptr)),
      traceFilePath_(getTaskTraceMetaFilePath(traceDir_)) {
  VELOX_CHECK_NOT_NULL(fs_);
  VELOX_CHECK(!fs_->exists(traceFilePath_));
}

void TaskTraceMetadataWriter::write(
    const core::QueryCtx& queryCtx,
    const core::PlanNode& planNode) {
  VELOX_CHECK(!finished_, "Query metadata can only be written once");
  finished_ = true;

  auto traceNode = trace::getTraceNode(planNode, traceNodeId_);

  const auto& credentialConfigKeys = queryCtx.credentialConfigKeys();

  folly::dynamic queryConfigObj = folly::dynamic::object;
  const auto configValues = queryCtx.queryConfig().rawConfigsCopy();
  for (const auto& [key, value] : configValues) {
    queryConfigObj[key] = redactIfCredential(credentialConfigKeys, key, value);
  }

  folly::dynamic connectorPropertiesObj = folly::dynamic::object;
  for (const auto& [connectorId, configs] :
       queryCtx.connectorSessionProperties()) {
    const auto credentialKeys = delegatedCredentialKeys(queryCtx, connectorId);
    folly::dynamic obj = folly::dynamic::object;
    for (const auto& [key, value] : configs->rawConfigsCopy()) {
      obj[key] = credentialKeys.contains(key)
          ? std::string(kRedactedConfigValue)
          : redactIfCredential(credentialConfigKeys, key, value);
    }
    connectorPropertiesObj[connectorId] = obj;
  }

  auto planNodeObj = traceNode->serialize();
  redactPlanCredentials(credentialConfigKeys, planNodeObj);

  folly::dynamic metaObj = folly::dynamic::object;
  metaObj[TraceTraits::kQueryConfigKey] = queryConfigObj;
  metaObj[TraceTraits::kConnectorPropertiesKey] = connectorPropertiesObj;
  metaObj[TraceTraits::kPlanNodeKey] = planNodeObj;

  const auto metaStr = folly::toJson(metaObj);
  const auto file = fs_->openFileForWrite(traceFilePath_);
  file->append(metaStr);
  file->close();
}

} // namespace facebook::velox::exec::trace
