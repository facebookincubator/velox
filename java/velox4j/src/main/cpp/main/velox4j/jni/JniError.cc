#include "JniError.h"
#include <velox/common/base/Exceptions.h>

namespace velox4j {
void JniErrorState::ensureInitialized(JNIEnv* env) {
  std::lock_guard<std::mutex> lockGuard(mtx_);
  if (initialized_) {
    return;
  }
  initialize(env);
  initialized_ = true;
}

void JniErrorState::assertInitialized() const {
  if (!initialized_) {
    VELOX_FAIL(
        "Fatal: JniErrorState::Initialize(...) was not called before "
        "using the utility");
  }
}

jclass JniErrorState::runtimeExceptionClass() {
  assertInitialized();
  return runtimeExceptionClass_;
}

jclass JniErrorState::illegalAccessExceptionClass() {
  assertInitialized();
  return illegalAccessExceptionClass_;
}

jclass JniErrorState::veloxExceptionClass() {
  assertInitialized();
  return veloxExceptionClass_;
}

void JniErrorState::initialize(JNIEnv* env) {
  veloxExceptionClass_ = createGlobalClassReference(
      env, "Lcom/meta/velox4j/exception/VeloxException;");
  ioExceptionClass_ = createGlobalClassReference(env, "Ljava/io/IOException;");
  runtimeExceptionClass_ =
      createGlobalClassReference(env, "Ljava/lang/RuntimeException;");
  unsupportedOperationExceptionClass_ = createGlobalClassReference(
      env, "Ljava/lang/UnsupportedOperationException;");
  illegalAccessExceptionClass_ =
      createGlobalClassReference(env, "Ljava/lang/IllegalAccessException;");
  illegalArgumentExceptionClass_ =
      createGlobalClassReference(env, "Ljava/lang/IllegalArgumentException;");
  JavaVM* vm;
  if (env->GetJavaVM(&vm) != JNI_OK) {
    VELOX_FAIL("Unable to get JavaVM instance");
  }
  vm_ = vm;
}

void JniErrorState::close() {
  std::lock_guard<std::mutex> lockGuard(mtx_);
  if (closed_) {
    return;
  }
  JNIEnv* env = getLocalJNIEnv();
  env->DeleteGlobalRef(veloxExceptionClass_);
  env->DeleteGlobalRef(ioExceptionClass_);
  env->DeleteGlobalRef(runtimeExceptionClass_);
  env->DeleteGlobalRef(unsupportedOperationExceptionClass_);
  env->DeleteGlobalRef(illegalAccessExceptionClass_);
  env->DeleteGlobalRef(illegalArgumentExceptionClass_);
  closed_ = true;
}

JniErrorState* getJniErrorState() {
  static JniErrorState jniErrorState;
  return &jniErrorState;
}
} // namespace velox4j
