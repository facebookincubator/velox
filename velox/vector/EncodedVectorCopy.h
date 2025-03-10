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

#include "velox/vector/BaseVector.h"

namespace facebook::velox {

struct EncodedVectorCopyOptions {
  memory::MemoryPool* pool;
  bool reuseSource;
};

/// Copy the vector while try to preserve the encoding on `target' (with
/// exceptions listed below).  If `target' is nullptr, preserve the encoding on
/// `source'.
///
/// `ranges' should not have any overlaps in target (overlaps in source are
/// allowed).  If target ranges exceeds the old target vector size, the vector
/// will be automatically extended; in this case, the target ranges must cover
/// all the missing part from the old vector.
///
/// In the following cases we do not preserve the exact encoding on `target':
/// - We merge multiple adjacent layers of dictionary and constant wrappers into
///   one.
/// - When `target' is constant, we convert it to dictionary to allow different
///   values in `source'.
/// - When `target' is flat ROW, MAP, or ARRAY, and `source' is constant or
///   dictionary encoded, the result will be dictionary encoded, to avoid
///   flattening the child vectors.  Once the target becomes dictionary, it can
///   stay that way and we can keep adding new content to it while keeping the
///   encoding, this is a typical use case for encoding preserved merging.
void encodedVectorCopy(
    const EncodedVectorCopyOptions& options,
    const VectorPtr& source,
    const folly::Range<const BaseVector::CopyRange*>& ranges,
    VectorPtr& target);

} // namespace facebook::velox
