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
#pragma once

#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY

#include "velox/exec/InMemoryExchangeClient.h"

namespace facebook::velox::exec {

/// Legacy name for InMemoryExchangeClient. Prefer InMemoryExchangeClient and
/// the header that declares it.
///
/// A type alias rather than a subclass, so a pre-migration
/// std::shared_ptr<ExchangeClient> parameter still overrides the
/// Operator::PlanNodeTranslator::toOperator virtual that now names
/// InMemoryExchangeClient. Retained, together with this header, so the
/// read-only-synced Prestissimo build keeps compiling. Its Buck targets define
/// VELOX_ENABLE_BACKWARD_COMPATIBILITY, while velox and open-source builds
/// never do.
///
/// velox/exec/Exchange.h includes this header, which is what puts the alias on
/// the include path those callers already take; declaring it here alone would
/// leave it unreachable. Remove that include together with this file once every
/// caller uses InMemoryExchangeClient.
using ExchangeClient = InMemoryExchangeClient;

} // namespace facebook::velox::exec

#endif // VELOX_ENABLE_BACKWARD_COMPATIBILITY
