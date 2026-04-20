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

// GPU shadow for folly/Likely.h
// Provides FOLLY_LIKELY / FOLLY_UNLIKELY macros.
// LIKELY / UNLIKELY are already defined in our CPortability.h shadow.
#pragma once

#include "folly/CPortability.h"

#ifndef FOLLY_LIKELY
#define FOLLY_LIKELY(x) (__builtin_expect((x), 1))
#endif

#ifndef FOLLY_UNLIKELY
#define FOLLY_UNLIKELY(x) (__builtin_expect((x), 0))
#endif
