#include "JniCommon.h"
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <execinfo.h>
#include <glog/logging.h>
#include <jni.h>
#include <velox/common/base/Exceptions.h>

namespace velox4j {

std::string jStringToCString(JNIEnv* env, jstring string) {
  int32_t jlen, clen;
  clen = env->GetStringUTFLength(string);
  jlen = env->GetStringLength(string);
  char buffer[clen];
  env->GetStringUTFRegion(string, 0, jlen, buffer);
  return {buffer, static_cast<uint64_t>(clen)};
}
void checkException(JNIEnv* env) {
  if (env->ExceptionCheck()) {
    jthrowable t = env->ExceptionOccurred();
    env->ExceptionClear();
    jclass describerClass =
        env->FindClass("com/meta/velox4j/exception/ExceptionDescriber");
    jmethodID describeMethod = env->GetStaticMethodID(
        describerClass,
        "describe",
        "(Ljava/lang/Throwable;)Ljava/lang/String;");
    std::string description = jStringToCString(
        env,
        (jstring)env->CallStaticObjectMethod(
            describerClass, describeMethod, t));
    if (env->ExceptionCheck()) {
      LOG(WARNING) << "Fatal: Uncaught Java exception during calling the Java "
                      "exception describer method! ";
    }
    VELOX_FAIL(
        "Error during calling Java code from native code: " + description);
  }
}

jclass createGlobalClassReference(JNIEnv* env, const char* className) {
  jclass localClass = env->FindClass(className);
  jclass globalClass = (jclass)env->NewGlobalRef(localClass);
  env->DeleteLocalRef(localClass);
  return globalClass;
}

jclass createGlobalClassReferenceOrError(JNIEnv* env, const char* className) {
  jclass globalClass = createGlobalClassReference(env, className);
  if (globalClass == nullptr) {
    std::string errorMessage =
        "Unable to CreateGlobalClassReferenceOrError for" +
        std::string(className);
    VELOX_FAIL(errorMessage);
  }
  return globalClass;
}

jmethodID
getMethodId(JNIEnv* env, jclass thisClass, const char* name, const char* sig) {
  jmethodID ret = env->GetMethodID(thisClass, name, sig);
  return ret;
}

jmethodID getMethodIdOrError(
    JNIEnv* env,
    jclass thisClass,
    const char* name,
    const char* sig) {
  jmethodID ret = getMethodId(env, thisClass, name, sig);
  if (ret == nullptr) {
    std::string errorMessage = "Unable to find method " + std::string(name) +
        " within signature" + std::string(sig);
    VELOX_FAIL(errorMessage);
  }
  return ret;
}
jmethodID getStaticMethodId(
    JNIEnv* env,
    jclass thisClass,
    const char* name,
    const char* sig) {
  jmethodID ret = env->GetStaticMethodID(thisClass, name, sig);
  return ret;
}
jmethodID getStaticMethodIdOrError(
    JNIEnv* env,
    jclass thisClass,
    const char* name,
    const char* sig) {
  jmethodID ret = getStaticMethodId(env, thisClass, name, sig);
  if (ret == nullptr) {
    std::string errorMessage = "Unable to find static method " +
        std::string(name) + " within signature" + std::string(sig);
    VELOX_FAIL(errorMessage);
  }
  return ret;
}

JNIEnv* getLocalJNIEnv() {
  static std::atomic<uint32_t> nextThreadId{0};
  if (spotify::jni::JavaThreadUtils::getEnvForCurrentThread() == nullptr) {
    const std::string threadName =
        fmt::format("Velox4J Native Thread {}", nextThreadId++);
    std::vector<char> threadNameCStr(threadName.length() + 1);
    std::strcpy(threadNameCStr.data(), threadName.data());
    JavaVM* vm = spotify::jni::JavaThreadUtils::getJavaVM();
    JNIEnv* env{nullptr};
    JavaVMAttachArgs args;
    args.version = JAVA_VERSION;
    args.name = threadNameCStr.data();
    args.group = nullptr;
    const int result =
        vm->AttachCurrentThreadAsDaemon(reinterpret_cast<void**>(&env), &args);
    if (result != JNI_OK) {
      VELOX_FAIL("Failed to reattach current thread to JVM.");
    }
    return env;
  }
  JNIEnv* env = spotify::jni::JavaThreadUtils::getEnvForCurrentThread();
  VELOX_CHECK(env != nullptr);
  return env;
}

template <typename T>
T* jniCastOrThrow(jlong handle) {
  auto instance = reinterpret_cast<T*>(handle);
  VELOX_CHECK(
      instance != nullptr, "FATAL: resource instance should not be null.");
  return instance;
}

spotify::jni::ClassRegistry* jniClassRegistry() {
  static spotify::jni::ClassRegistry gClasses;
  return &gClasses;
}
} // namespace velox4j
