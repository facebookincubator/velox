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

#ifdef _WIN32
// Ensure target architecture is defined for Windows SDK headers
// MSBuild normally sets this, but NuGet-based toolsets may not
#if defined(_M_AMD64) && !defined(_AMD64_)
#define _AMD64_
#endif

// Prevent Windows min/max macros from conflicting with std::min/std::max
#ifndef NOMINMAX
#define NOMINMAX
#endif

// Reduce Windows header bloat
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Windows Socket Fix:
// Folly's NetOps.h defines sockaddr_un, and Windows SDK's afunix.h also
// tries to define it, causing a redefinition error.
// We prevent afunix.h from being included by defining its include guard.
#ifndef __AFUNIX_H__
#define __AFUNIX_H__
#endif
#endif  // _WIN32
