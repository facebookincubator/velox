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

// Guarded the same way as JavaRegex.h: clang-tidy scans diff-changed headers
// in isolation and cannot find <jni.h> on hosts without a JDK.
#if VELOX_REGEX_COMPAT_HAS_JAVA

#include <jni.h>

namespace facebook::velox::regex_compat {

/// Process-singleton embedded JVM used by the regex-compat test suite's
/// JavaRegex backend.  Boots the JVM on first `instance()` call via
/// `JNI_CreateJavaVM` and keeps it alive for the lifetime of the process —
/// JNI forbids destroy+recreate in the same process, so we never tear down.
///
/// Tests should register this as a GTest GlobalEnvironment via
/// JvmFixture::Register() in main(), to give the JVM boot a clear lifecycle
/// boundary distinct from per-test setup.
class JvmFixture {
 public:
  static JvmFixture& instance();

  JavaVM* jvm() const { return jvm_; }
  JNIEnv* env() const { return env_; }

  /// Register this fixture as a GTest GlobalEnvironment.  Call from main().
  static void Register();

 private:
  JvmFixture();
  ~JvmFixture() = default;

  JavaVM* jvm_ = nullptr;
  JNIEnv* env_ = nullptr;
};

} // namespace facebook::velox::regex_compat

#endif // VELOX_REGEX_COMPAT_HAS_JAVA
