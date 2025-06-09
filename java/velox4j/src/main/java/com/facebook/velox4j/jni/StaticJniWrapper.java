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
package com.facebook.velox4j.jni;

import com.facebook.velox4j.memory.AllocationListener;

/**
 * A static JNI wrapper that is independent to any JNI sessions. All the JNI methods defined in the
 * static JNI wrapper are globally available without having to create a session first.
 */
public class StaticJniWrapper {
  private static final StaticJniWrapper INSTANCE = new StaticJniWrapper();

  static StaticJniWrapper get() {
    return INSTANCE;
  }

  private StaticJniWrapper() {}

  // Global initialization.
  native void initialize(String globalConfJson);

  // Memory.
  native long createMemoryManager(AllocationListener listener);

  // Lifecycle.
  native long createSession(long memoryManagerId);

  native void releaseCppObject(long objectId);

  // For UpIterator.
  native int upIteratorAdvance(long id);

  native void upIteratorWait(long id);

  // For DownIterator.
  native void blockingQueuePut(long id, long rvId);

  native void blockingQueueNoMoreInput(long id);

  // For SerialTask.
  native void serialTaskAddSplit(
      long id, String planNodeId, int groupId, String connectorSplitJson);

  native void serialTaskNoMoreSplits(long id, String planNodeId);

  native String serialTaskCollectStats(long id);

  // For Variant.
  native String variantInferType(String json);

  // For BaseVector / RowVector / SelectivityVector.
  native void baseVectorToArrow(long rvid, long cSchema, long cArray);

  native String baseVectorSerialize(long[] id);

  native String baseVectorGetType(long id);

  native int baseVectorGetSize(long id);

  native String baseVectorGetEncoding(long id);

  native void baseVectorAppend(long id, long toAppendId);

  native boolean selectivityVectorIsValid(long id, int idx);

  // For TableWrite.
  native String tableWriteTraitsOutputType();

  // For serde.
  native String iSerializableAsJava(long id);

  native String variantAsJava(long id);
}
