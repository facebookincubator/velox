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
package com.facebook.velox4j.query;

import com.facebook.velox4j.connector.ConnectorSplit;
import com.facebook.velox4j.data.RowVector;
import com.facebook.velox4j.iterator.UpIterator;
import com.facebook.velox4j.jni.JniApi;
import com.facebook.velox4j.jni.StaticJniApi;

/**
 * An up-iterator implementation that is backed by a Velox task that runs in serial execution mode.
 */
public class SerialTask implements UpIterator {
  private final JniApi jniApi;
  private final long id;

  public SerialTask(JniApi jniApi, long id) {
    this.jniApi = jniApi;
    this.id = id;
  }

  @Override
  public State advance() {
    return StaticJniApi.get().upIteratorAdvance(this);
  }

  @Override
  public void waitFor() {
    StaticJniApi.get().upIteratorWait(this);
  }

  @Override
  public RowVector get() {
    return jniApi.upIteratorGet(this);
  }

  @Override
  public long id() {
    return id;
  }

  public void addSplit(String planNodeId, ConnectorSplit split) {
    StaticJniApi.get().serialTaskAddSplit(this, planNodeId, -1, split);
  }

  public void noMoreSplits(String planNodeId) {
    StaticJniApi.get().serialTaskNoMoreSplits(this, planNodeId);
  }

  public SerialTaskStats collectStats() {
    return StaticJniApi.get().serialTaskCollectStats(this);
  }
}
