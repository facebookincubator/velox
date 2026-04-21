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

// GPU shadow for velox/common/base/Exceptions.h
// Provides no-op stubs for VELOX_CHECK / VELOX_FAIL macros.
// On GPU, these become no-ops (call() bodies should not throw).
// Future: replace with __trap() for debug builds.
#pragma once

#ifndef VELOX_CHECK
#define VELOX_CHECK(...)
#endif

#ifndef VELOX_CHECK_EQ
#define VELOX_CHECK_EQ(...)
#endif

#ifndef VELOX_CHECK_NE
#define VELOX_CHECK_NE(...)
#endif

#ifndef VELOX_CHECK_LT
#define VELOX_CHECK_LT(...)
#endif

#ifndef VELOX_CHECK_LE
#define VELOX_CHECK_LE(...)
#endif

#ifndef VELOX_CHECK_GT
#define VELOX_CHECK_GT(...)
#endif

#ifndef VELOX_CHECK_GE
#define VELOX_CHECK_GE(...)
#endif

#ifndef VELOX_CHECK_NOT_NULL
#define VELOX_CHECK_NOT_NULL(...)
#endif

#ifndef VELOX_CHECK_NULL
#define VELOX_CHECK_NULL(...)
#endif

#ifndef VELOX_FAIL
#define VELOX_FAIL(...) ((void)0)
#endif

#ifndef VELOX_UNREACHABLE
#define VELOX_UNREACHABLE(...) __builtin_unreachable()
#endif

#ifndef VELOX_USER_CHECK
#define VELOX_USER_CHECK(...)
#endif

#ifndef VELOX_USER_CHECK_EQ
#define VELOX_USER_CHECK_EQ(...)
#endif

#ifndef VELOX_USER_CHECK_NE
#define VELOX_USER_CHECK_NE(...)
#endif

#ifndef VELOX_USER_CHECK_LT
#define VELOX_USER_CHECK_LT(...)
#endif

#ifndef VELOX_USER_CHECK_LE
#define VELOX_USER_CHECK_LE(...)
#endif

#ifndef VELOX_USER_CHECK_GT
#define VELOX_USER_CHECK_GT(...)
#endif

#ifndef VELOX_USER_CHECK_GE
#define VELOX_USER_CHECK_GE(...)
#endif

#ifndef VELOX_USER_CHECK_NOT_NULL
#define VELOX_USER_CHECK_NOT_NULL(...)
#endif

#ifndef VELOX_USER_FAIL
#define VELOX_USER_FAIL(...) ((void)0)
#endif

#ifndef VELOX_DCHECK
#define VELOX_DCHECK(...)
#endif

#ifndef VELOX_DCHECK_EQ
#define VELOX_DCHECK_EQ(...)
#endif

#ifndef VELOX_DCHECK_NE
#define VELOX_DCHECK_NE(...)
#endif

#ifndef VELOX_DCHECK_LT
#define VELOX_DCHECK_LT(...)
#endif

#ifndef VELOX_DCHECK_LE
#define VELOX_DCHECK_LE(...)
#endif

#ifndef VELOX_DCHECK_GT
#define VELOX_DCHECK_GT(...)
#endif

#ifndef VELOX_DCHECK_GE
#define VELOX_DCHECK_GE(...)
#endif

#ifndef VELOX_DCHECK_NOT_NULL
#define VELOX_DCHECK_NOT_NULL(...)
#endif

#ifndef VELOX_NYI
#define VELOX_NYI(...) ((void)0)
#endif

#ifndef VELOX_UNSUPPORTED
#define VELOX_UNSUPPORTED(...) ((void)0)
#endif

#ifndef VELOX_ARITHMETIC_ERROR
#define VELOX_ARITHMETIC_ERROR(...) ((void)0)
#endif
