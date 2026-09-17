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

namespace facebook::velox::ucx_exchange {

/// Registers the UCX transport on both sides of an exchange edge under
/// core::TransportKind::kUcx: the output queue manager paired with the
/// UcxPartitionedOutput builder, and the UcxExchangeClient paired with the
/// UcxExchange builder. Pairing an operator builder with its manager or client
/// in one entry is what guarantees the two halves cannot diverge. Idempotent:
/// a second call replaces the existing entries.
///
/// This is process-level capability only. Which nodes speak UCX is a pure plan
/// property -- PartitionedOutputNode::transportKind() and
/// ExchangeNode::transportKind(), resolved by exec::Task -- so this
/// registration must never be consulted for per-node operator selection. A plan
/// that names kUcx on a worker where UCX exchange is disabled fails fast in
/// exec::Task rather than silently downgrading to the in-memory transport; the
/// coordinator is responsible for only naming kUcx where the workers register
/// it.
void registerUcxTransports();

/// Removes both kUcx registrations, undoing registerUcxTransports(). Leaves
/// every other transport alone, including the built-in in-memory default.
/// Idempotent, and a no-op when the transport was never registered.
void unregisterUcxTransports();

} // namespace facebook::velox::ucx_exchange
