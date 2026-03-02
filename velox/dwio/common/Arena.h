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

#include <google/protobuf/arena.h>
#include <google/protobuf/stubs/common.h>

namespace facebook::velox::dwio::common {

/// Wrapper over protobuf's arena allocation.  The API changes from
/// CreateMessage() to Create() in newer protobuf versions.
template <class T, class... Args>
T* ArenaCreate(google::protobuf::Arena* arena, Args&&... args) {
#if GOOGLE_PROTOBUF_VERSION >= 5030000
  return google::protobuf::Arena::Create<T>(arena, std::forward<Args>(args)...);
#else
  return google::protobuf::Arena::CreateMessage<T>(
      arena, std::forward<Args>(args)...);
#endif
}

} // namespace facebook::velox::dwio::common
