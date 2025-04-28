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
package com.meta.velox4j.jni;

import com.google.common.annotations.VisibleForTesting;

import com.meta.velox4j.iterator.DownIterator;

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
