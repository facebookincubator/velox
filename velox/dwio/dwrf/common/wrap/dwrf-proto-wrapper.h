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

#include "velox/dwio/common/Adaptor.h"

DIAGNOSTIC_PUSH
DIAGNOSTIC_IGNORE("-Wconversion")
DIAGNOSTIC_IGNORE("-Wdeprecated")
DIAGNOSTIC_IGNORE("-Wsign-conversion")
DIAGNOSTIC_IGNORE("-Wunused-parameter")

#ifdef __clang__
DIAGNOSTIC_IGNORE("-Wnested-anon-types")
DIAGNOSTIC_IGNORE("-Wreserved-id-macro")
DIAGNOSTIC_IGNORE("-Wshorten-64-to-32")
DIAGNOSTIC_IGNORE("-Wweak-vtables")
#endif

// Prevent Velox type function names from conflicting with protobuf enum values
// The protobuf has enum values like VOID, INT, etc. that conflict with
// Velox's VOID(), INT() type factory functions
#ifdef _MSC_VER
#pragma push_macro("VOID")
#pragma push_macro("BOOLEAN")
#pragma push_macro("BYTE")
#pragma push_macro("SHORT")
#pragma push_macro("INT")
#pragma push_macro("LONG")
#pragma push_macro("FLOAT")
#pragma push_macro("DOUBLE")
#pragma push_macro("STRING")
#pragma push_macro("DATE")
#pragma push_macro("TIMESTAMP")
#pragma push_macro("BINARY")
#pragma push_macro("DECIMAL")
#undef VOID
#undef BOOLEAN
#undef BYTE
#undef SHORT
#undef INT
#undef LONG
#undef FLOAT
#undef DOUBLE
#undef STRING
#undef DATE
#undef TIMESTAMP
#undef BINARY
#undef DECIMAL
#endif

#include "velox/dwio/dwrf/proto/dwrf_proto.pb.h"

#ifdef _MSC_VER
#pragma pop_macro("DECIMAL")
#pragma pop_macro("BINARY")
#pragma pop_macro("TIMESTAMP")
#pragma pop_macro("DATE")
#pragma pop_macro("STRING")
#pragma pop_macro("DOUBLE")
#pragma pop_macro("FLOAT")
#pragma pop_macro("LONG")
#pragma pop_macro("INT")
#pragma pop_macro("SHORT")
#pragma pop_macro("BYTE")
#pragma pop_macro("BOOLEAN")
#pragma pop_macro("VOID")
#endif

DIAGNOSTIC_POP
