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

// GPU shadow for velox/type/SimpleFunctionApi.h.
//
// The real header is two things stacked: the type tags a call() body names,
// and the function-signature machinery (FunctionSignature, SignatureBuilder,
// the simple-function reflection utilities) that is host-only. Only the first
// half is wanted here, and it cannot be reached without the second.
//
// The tags are not restated. SimpleFunctionTags.h holds them and parses under
// nvcc on its own, so it is included directly -- one definition of Date and
// Generic<> rather than a copy that can drift from it.
#pragma once

#include "velox/type/SimpleFunctionTags.h"
