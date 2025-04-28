#include "JavaAllocationListener.h"
#include <glog/logging.h>
#include "velox4j/jni/JniCommon.h"

namespace velox4j {

namespace {
const char* kClassName = "com/meta/velox4j/memory/AllocationListener";
}
const char* JavaAllocationListenerJniWrapper::getCanonicalName() const {
  return kClassName;
}

void JavaAllocationListenerJniWrapper::initialize(JNIEnv* env) {
  JavaClass::setClass(env);

  cacheMethod(env, "allocationChanged", kTypeVoid, kTypeLong, nullptr);

  registerNativeMethods(env);
}

void JavaAllocationListenerJniWrapper::mapFields() {}

JavaAllocationListener::JavaAllocationListener(JNIEnv* env, jobject ref) {
  ref_ = env->NewGlobalRef(ref);
}

JavaAllocationListener::~JavaAllocationListener() {
  try {
    getLocalJNIEnv()->DeleteGlobalRef(ref_);
  } catch (const std::exception& ex) {
    LOG(WARNING) << "Unable to destroy the global reference to the Java side "
                    "allocation listener: "
                 << ex.what();
  }
}

void JavaAllocationListener::allocationChanged(int64_t diff) {
  static const auto* clazz = jniClassRegistry()->get(kClassName);
  static jmethodID methodId = clazz->getMethod("allocationChanged");
  if (diff == 0) {
    return;
  }
  JNIEnv* env = getLocalJNIEnv();
  env->CallLongMethod(ref_, methodId, diff);
  usedBytes_ += diff;
  while (true) {
    int64_t savedPeakBytes = peakBytes_;
    if (usedBytes_ <= savedPeakBytes) {
      break;
    }
    // usedBytes_ > savedPeakBytes, update peak
    if (peakBytes_.compare_exchange_weak(savedPeakBytes, usedBytes_)) {
      break;
    }
  }
}

const int64_t JavaAllocationListener::currentBytes() const {
  return usedBytes_;
}

const int64_t JavaAllocationListener::peakBytes() const {
  return peakBytes_;
}
} // namespace velox4j
