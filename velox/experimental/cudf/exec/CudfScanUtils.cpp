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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConnector.h"
#include "velox/experimental/cudf/exec/CudfScanUtils.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/TableHandle.h"

namespace facebook::velox::cudf_velox {
namespace {

bool reject(std::string* reason, std::string message) {
  if (reason != nullptr) {
    *reason = std::move(message);
  }
  return false;
}

// Returns whether legacy_cast leaves a cast from 'source' to 'target' as is: a
// cast between BOOLEAN, integer, REAL, DOUBLE, DECIMAL and DATE types, or from
// VARCHAR to DATE or DECIMAL.
// NOTE: The list is short on purpose. legacy_cast changes how a VARCHAR parses
// as a TIMESTAMP, how a TIMESTAMP, REAL or DOUBLE prints as a VARCHAR, and
// whether a VARCHAR with non-ASCII characters casts to an integer.
bool isLegacyCastIndependent(const TypePtr& source, const TypePtr& target) {
  const auto isNumberOrDate = [](const TypePtr& type) {
    return type->isDate() || type->isDecimal() ||
        (std::string_view(type->name()) == type->kindName() &&
         (type->isBoolean() || type->isTinyint() || type->isSmallint() ||
          type->isInteger() || type->isBigint() || type->isReal() ||
          type->isDouble()));
  };
  if (source->isVarchar()) {
    return target->isDate() || target->isDecimal();
  }
  return isNumberOrDate(source) && isNumberOrDate(target);
}

// Returns whether 'expr' holds a CAST or TRY_CAST that legacy_cast may change.
bool containsLegacyCastDependentCast(const core::TypedExprPtr& expr) {
  if (expr->isCastKind() &&
      !isLegacyCastIndependent(expr->inputs().at(0)->type(), expr->type())) {
    return true;
  }
  return std::any_of(
      expr->inputs().begin(), expr->inputs().end(), [](const auto& input) {
        return containsLegacyCastDependentCast(input);
      });
}

} // namespace

bool isGpuTableScan(
    const core::TableScanNode& node,
    const velox::connector::ConnectorQueryCtx& connectorQueryCtx,
    const core::QueryConfig& queryConfig,
    std::string* reason) {
  auto scanConnector = velox::connector::ConnectorRegistry::tryGet(
      node.tableHandle()->connectorId());
  if (dynamic_cast<const connector::hive::CudfHiveConnector*>(
          scanConnector.get()) == nullptr &&
      dynamic_cast<const connector::hive::iceberg::CudfIcebergConnector*>(
          scanConnector.get()) == nullptr) {
    return reject(
        reason,
        "TableScan connector is not CudfHiveConnector or CudfIcebergConnector");
  }
  if (!isTypeSupportedByCudf(node.outputType())) {
    return reject(
        reason,
        fmt::format(
            "TableScan output type is unsupported by cuDF: {}",
            node.outputType()->toString()));
  }
  if (const auto unsupported =
          connector::hive::CudfHiveDataSource::unsupportedReason(
              node.tableHandle(), connectorQueryCtx)) {
    return reject(
        reason,
        fmt::format(
            "TableScan is not supported by the cuDF data source: {}",
            *unsupported));
  }

  // TODO: Remove this check once the data source folds with the session
  // QueryConfig (facebookincubator/velox#19476).
  const auto hiveTableHandle =
      std::dynamic_pointer_cast<const velox::connector::hive::HiveTableHandle>(
          node.tableHandle());
  VELOX_CHECK_NOT_NULL(hiveTableHandle);
  const auto& remainingFilter = hiveTableHandle->remainingFilter();
  if (queryConfig.isLegacyCast() && remainingFilter &&
      containsLegacyCastDependentCast(remainingFilter)) {
    VELOX_UNSUPPORTED(
        "Remaining filter with a cast that legacy_cast may change is not supported by the cuDF data source: {}",
        remainingFilter->toString());
  }
  return true;
}

} // namespace facebook::velox::cudf_velox
