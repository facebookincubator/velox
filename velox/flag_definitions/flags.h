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

/// When GFlags are used, they must be translated to
/// velox::GlobalConfig by invoking translateFlagsToGlobalConfig
namespace facebook::velox {
void translateFlagsToGlobalConfig();
}

/// Use O_DIRECT for SSD cache I/O. This allows to bypass Linux Kernel's page
/// cache and can improve performance on some filesystems. Disable if the
/// filesystem does not support it.
DECLARE_bool(velox_ssd_odirect);

/// Verify the data written to SSD. Once an entry is written, it is immediately
/// read back and is compared against the entry written.
/// This is helpful to protect against SSD write corruption.
DECLARE_bool(velox_ssd_verify_write);

/// Enable the stacktrace for user type of VeloxException
DECLARE_bool(velox_exception_user_stacktrace_enabled);

/// Enable the stacktrace for system type of VeloxException
DECLARE_bool(velox_exception_system_stacktrace_enabled);

/// Min time interval in milliseconds between stack traces captured in
/// user type of VeloxException; off when set to 0 (the default)
DECLARE_int32(velox_exception_user_stacktrace_rate_limit_ms);

/// Min time interval in milliseconds between stack traces captured in
/// system type of VeloxException; off when set to 0 (the default)
DECLARE_int32(velox_exception_system_stacktrace_rate_limit_ms);
