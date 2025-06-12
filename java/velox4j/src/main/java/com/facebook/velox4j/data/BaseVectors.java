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
package com.facebook.velox4j.data;

import java.util.List;

import com.google.common.base.Preconditions;

import com.facebook.velox4j.jni.JniApi;
import com.facebook.velox4j.jni.StaticJniApi;
import com.facebook.velox4j.type.Type;

public class BaseVectors {
  private final JniApi jniApi;

  public BaseVectors(JniApi jniApi) {
    this.jniApi = jniApi;
  }

  public BaseVector createEmpty(Type type) {
    return jniApi.createEmptyBaseVector(type);
  }

  public static String serializeOne(BaseVector vector) {
    return StaticJniApi.get().baseVectorSerialize(List.of(vector));
  }

  public BaseVector deserializeOne(String serialized) {
    final List<BaseVector> vectors = jniApi.baseVectorDeserialize(serialized);
    Preconditions.checkState(
        vectors.size() == 1, "Expected one vector, but got %s", vectors.size());
    return vectors.get(0);
  }

  public static String serializeAll(List<? extends BaseVector> vectors) {
    return StaticJniApi.get().baseVectorSerialize(vectors);
  }

  public List<BaseVector> deserializeAll(String serialized) {
    return jniApi.baseVectorDeserialize(serialized);
  }
}
