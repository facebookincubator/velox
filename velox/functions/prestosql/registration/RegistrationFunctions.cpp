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
#include <string>
#include "velox/functions/prestosql/IPAddressFunctions.h"
#include "velox/functions/prestosql/UuidFunctions.h"
#include "velox/functions/prestosql/types/P4HyperLogLogRegistration.h"

namespace facebook::velox::functions {

extern void registerMathematicalFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerMathematicalOperators(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerProbabilityTrigonometryFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerArrayFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerBitwiseFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerCheckedArithmeticFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerComparisonFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerDateTimeFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerGeneralFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerHyperLogFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerKHyperLogLogFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerTDigestFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerQDigestFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerSetDigestFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerSfmSketchFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerEnumFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerIntegerFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerFloatingPointFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerJsonFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerMapFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerStringFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerBinaryFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerURLFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerDataSizeFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerMapAllowingDuplicates(
    const std::string& name,
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerBingTileFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
#ifdef VELOX_ENABLE_GEO
extern void registerGeometryFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerSphericalGeographyFunctions();
#endif
#ifdef VELOX_ENABLE_GEO
extern void registerS2Functions(
    const std::string& prefix,
    std::string_view defaultOwner);
#endif
extern void registerInternalArrayFunctions(std::string_view defaultOwner);

namespace prestosql {
void registerArithmeticFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerMathematicalOperators(prefix, defaultOwner);
  functions::registerMathematicalFunctions(prefix, defaultOwner);
  functions::registerProbabilityTrigonometryFunctions(prefix, defaultOwner);
}

void registerCheckedArithmeticFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerCheckedArithmeticFunctions(prefix, defaultOwner);
}

void registerComparisonFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerComparisonFunctions(prefix, defaultOwner);
}

void registerArrayFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerArrayFunctions(prefix, defaultOwner);
}

void registerMapFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerMapFunctions(prefix, defaultOwner);
}

void registerJsonFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerJsonFunctions(prefix, defaultOwner);
}

void registerHyperLogFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerHyperLogFunctions(prefix, defaultOwner);
}

void registerTDigestFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerTDigestFunctions(prefix, defaultOwner);
}

void registerQDigestFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerQDigestFunctions(prefix, defaultOwner);
}

void registerSfmSketchFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerSfmSketchFunctions(prefix, defaultOwner);
}

void registerEnumFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerEnumFunctions(prefix, defaultOwner);
}

void registerIntegerFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerIntegerFunctions(prefix, defaultOwner);
}

void registerFloatingPointFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerFloatingPointFunctions(prefix, defaultOwner);
}

void registerBingTileFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerBingTileFunctions(prefix, defaultOwner);
}

#ifdef VELOX_ENABLE_GEO
void registerGeometryFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerGeometryFunctions(prefix, defaultOwner);
}

void registerSphericalGeographyFunctions() {
  functions::registerSphericalGeographyFunctions();
}

void registerS2Functions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerS2Functions(prefix, defaultOwner);
}
#endif

void registerGeneralFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerGeneralFunctions(prefix, defaultOwner);
}

void registerDateTimeFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerDateTimeFunctions(prefix, defaultOwner);
}

void registerURLFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerURLFunctions(prefix, defaultOwner);
}

void registerStringFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerStringFunctions(prefix, defaultOwner);
}

void registerBinaryFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerBinaryFunctions(prefix, defaultOwner);
}

void registerBitwiseFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerBitwiseFunctions(prefix, defaultOwner);
}

void registerAllScalarFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerP4HyperLogLogType();
  registerArithmeticFunctions(prefix, defaultOwner);
  registerCheckedArithmeticFunctions(prefix, defaultOwner);
  registerComparisonFunctions(prefix, defaultOwner);
  registerMapFunctions(prefix, defaultOwner);
  registerArrayFunctions(prefix, defaultOwner);
  registerJsonFunctions(prefix, defaultOwner);
  registerHyperLogFunctions(prefix, defaultOwner);
  registerKHyperLogLogFunctions(prefix, defaultOwner);
  registerTDigestFunctions(prefix, defaultOwner);
  registerQDigestFunctions(prefix, defaultOwner);
  registerSfmSketchFunctions(prefix, defaultOwner);
  registerSetDigestFunctions(prefix, defaultOwner);
  registerEnumFunctions(prefix, defaultOwner);
  registerIntegerFunctions(prefix, defaultOwner);
  registerFloatingPointFunctions(prefix, defaultOwner);
  registerBingTileFunctions(prefix, defaultOwner);
#ifdef VELOX_ENABLE_GEO
  registerGeometryFunctions(prefix, defaultOwner);
  registerSphericalGeographyFunctions();
  registerS2Functions(prefix, defaultOwner);
#endif
  registerGeneralFunctions(prefix, defaultOwner);
  registerDateTimeFunctions(prefix, defaultOwner);
  registerURLFunctions(prefix, defaultOwner);
  registerStringFunctions(prefix, defaultOwner);
  registerBinaryFunctions(prefix, defaultOwner);
  registerBitwiseFunctions(prefix, defaultOwner);
  registerUuidFunctions(prefix, defaultOwner);
  registerIPAddressFunctions(prefix, defaultOwner);
  registerDataSizeFunctions(prefix, defaultOwner);
}

void registerMapAllowingDuplicates(
    const std::string& name,
    const std::string& prefix,
    std::string_view defaultOwner) {
  functions::registerMapAllowingDuplicates(name, prefix, defaultOwner);
}

void registerInternalFunctions(std::string_view defaultOwner) {
  functions::registerInternalArrayFunctions(defaultOwner);
}
} // namespace prestosql

} // namespace facebook::velox::functions
