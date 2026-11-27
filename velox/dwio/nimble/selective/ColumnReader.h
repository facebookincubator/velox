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

#pragma once

#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/nimble/selective/NimbleData.h"

namespace facebook::nimble {

std::unique_ptr<velox::dwio::common::SelectiveColumnReader> buildColumnReader(
    const velox::TypePtr& requestedType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    velox::common::ScanSpec& scanSpec,
    bool isRoot);

/// Rebinds 'reader', which buildColumnReader() built for an earlier stripe, to
/// the stripe that 'params' reads, through SelectiveColumnReader::rebind().
/// 'params' describes the column 'reader' was built for. The caller first
/// moves the StripeStreams that 'params' and 'reader' share to that stripe with
/// StripeStreams::setStripe(), and loads the streams that the rebind enqueues
/// with StripeStreams::load() after it. Only the decoders of the integer,
/// floating-point and list readers then still point at an earlier stripe, or
/// at no stream after SelectiveColumnReader::unbind(), so rebinding gives each
/// decoder the same stream of the new stripe. Returns false, having enqueued no
/// stream, if 'reader' cannot read that stripe, for example because a stream it
/// reads is absent there, in which case the caller builds a new reader for it.
bool rebindColumnReader(
    velox::dwio::common::SelectiveColumnReader& reader,
    NimbleParams& params);

} // namespace facebook::nimble
