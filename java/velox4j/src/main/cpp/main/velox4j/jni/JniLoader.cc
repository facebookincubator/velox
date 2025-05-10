#include <JniHelpers.h>
#include <glog/logging.h>
#include <jni.h>
#include "JniWrapper.h"
#include "StaticJniWrapper.h"
#include "velox4j/iterator/DownIterator.h"
#include "velox4j/jni/JniCommon.h"
#include "velox4j/jni/JniError.h"
#include "velox4j/memory/JavaAllocationListener.h"

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* jvm, void*) {
  LOG(INFO) << "Initializing Velox4J...";
  JNIEnv* env = jniHelpersInitialize(jvm);
  if (env == nullptr) {
    return -1;
  }

  velox4j::getJniErrorState()->ensureInitialized(env);
  velox4j::jniClassRegistry()->add(env, new velox4j::StaticJniWrapper(env));
  velox4j::jniClassRegistry()->add(env, new velox4j::JniWrapper(env));
  velox4j::jniClassRegistry()->add(
      env, new velox4j::DownIteratorJniWrapper(env));
  velox4j::jniClassRegistry()->add(
      env, new velox4j::JavaAllocationListenerJniWrapper(env));

  LOG(INFO) << "Velox4J initialized.";
  return JAVA_VERSION;
}
