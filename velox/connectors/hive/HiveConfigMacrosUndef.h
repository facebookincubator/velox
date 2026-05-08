// NOLINT(facebook-hte-MultipleIncludeGuardMissing)
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

// Undefines macros from HiveConfigMacrosDefine.h.
// Must be included after all macro usage.
// No #pragma once — see HiveConfigMacrosDefine.h for explanation.

#ifndef VELOX_HIVE_CONFIG_MACROS_DEFINED
#error "HiveConfigMacrosUndef.h included without HiveConfigMacrosDefine.h"
#endif
#undef VELOX_HIVE_CONFIG_MACROS_DEFINED

#undef VELOX_HIVE_CONFIG
#undef VELOX_HIVE_CONFIG_LEGACY
#undef VELOX_HIVE_CONFIG_PROPERTY
