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
#include "velox/external/regex_compat/JvmFixture.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace facebook::velox::regex_compat {
namespace {

class JvmGlobalEnv : public ::testing::Environment {
 public:
  void SetUp() override {
    // Force JVM construction now (before any test runs).
    JvmFixture::instance();
  }
  // No TearDown: JNI forbids JVM destroy + recreate in the same process.
};

} // namespace

JvmFixture::JvmFixture() {
  JavaVMInitArgs args{};
  args.version = JNI_VERSION_1_8;
  args.ignoreUnrecognized = JNI_FALSE;
  args.nOptions = 0;
  args.options = nullptr;

  const jint rc =
      JNI_CreateJavaVM(&jvm_, reinterpret_cast<void**>(&env_), &args);
  if (rc != JNI_OK) {
    std::ostringstream os;
    os << "JvmFixture: JNI_CreateJavaVM failed with code " << rc;
    throw std::runtime_error(os.str());
  }
}

JvmFixture& JvmFixture::instance() {
  // Function-local static guarantees thread-safe one-time construction
  // (C++11+) and avoids static-init order issues.
  static JvmFixture inst;
  return inst;
}

void JvmFixture::Register() {
  ::testing::AddGlobalTestEnvironment(new JvmGlobalEnv);
}

} // namespace facebook::velox::regex_compat
