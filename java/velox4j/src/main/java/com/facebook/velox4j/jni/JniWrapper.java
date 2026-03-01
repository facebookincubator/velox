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

import com.google.common.annotations.VisibleForTesting;

import com.facebook.velox4j.iterator.DownIterator;

/**
 * A dynamic JniWrapper that includes the JNI methods that are session-aware. Which means, the
 * sanity of these methods usually rely on certain objects that were stored in the current session.
 * For example, an API that turns a Velox vector into another, then returns it to Java - this method
 * will read and write objects from and to the current JNI session storage. So the method will be
 * defined in the (dynamic) JniWrapper.
 */
final class JniWrapper {
  private final long sessionId;

  JniWrapper(long sessionId) {
    this.sessionId = sessionId;
  }

  @CalledFromNative
  public long sessionId() {
    return sessionId;
  }

  // Expression evaluation.
  native long createEvaluator(String evalJson);

  native long evaluatorEval(long evaluatorId, long selectivityVectorId, long rvId);

  // Plan execution.
  native long createQueryExecutor(String queryJson);

  native long queryExecutorExecute(long id);

  // For UpIterator.
  native long upIteratorGet(long id);

  // For DownIterator.
  native long createExternalStreamFromDownIterator(DownIterator itr);

  native long createBlockingQueue();

  // For BaseVector / RowVector / SelectivityVector.
  native long createEmptyBaseVector(String typeJson);

  native long arrowToBaseVector(long cSchema, long cArray);

  native long[] baseVectorDeserialize(String serialized);

  native long baseVectorWrapInConstant(long id, int length, int index);

  native long baseVectorSlice(long id, int offset, int length);

  native long baseVectorLoadedVector(long id);

  native long createSelectivityVector(int length);

  // For TableWrite.
  native String tableWriteTraitsOutputTypeWithAggregationNode(String aggregationNodeJson);

  // For serde.
  native long iSerializableAsCpp(String json);

  native long variantAsCpp(String json);

  @VisibleForTesting
  native long createUpIteratorWithExternalStream(long id);
}
