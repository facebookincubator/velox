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

import com.google.common.base.Preconditions;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;

import com.facebook.velox4j.arrow.Arrow;
import com.facebook.velox4j.exception.VeloxException;
import com.facebook.velox4j.jni.CppObject;
import com.facebook.velox4j.jni.JniApi;
import com.facebook.velox4j.jni.StaticJniApi;
import com.facebook.velox4j.type.Type;

public class BaseVector implements CppObject {
  public static BaseVector wrap(JniApi jniApi, long id, VectorEncoding encoding) {
    // TODO Add JNI API `isRowVector` for performance.
    if (encoding == VectorEncoding.ROW) {
      return new RowVector(jniApi, id);
    }
    return new BaseVector(jniApi, id, encoding);
  }

  private final JniApi jniApi;
  private final long id;
  private final VectorEncoding encoding;

  protected BaseVector(JniApi jniApi, long id, VectorEncoding encoding) {
    this.jniApi = jniApi;
    this.id = id;
    this.encoding = encoding;
  }

  @Override
  public long id() {
    return id;
  }

  public Type getType() {
    return StaticJniApi.get().baseVectorGetType(this);
  }

  public VectorEncoding getEncoding() {
    return encoding;
  }

  public int getSize() {
    return StaticJniApi.get().baseVectorGetSize(this);
  }

  public BaseVector wrapInConstant(int length, int index) {
    return jniApi.baseVectorWrapInConstant(this, length, index);
  }

  public BaseVector slice(int offset, int length) {
    return jniApi.baseVectorSlice(this, offset, length);
  }

  public void append(BaseVector toAppend) {
    StaticJniApi.get().baseVectorAppend(this, toAppend);
  }

  public BaseVector loadedVector() {
    return jniApi.loadedVector(this);
  }

  public String serialize() {
    return BaseVectors.serializeOne(this);
  }

  public String toString(BufferAllocator alloc) {
    try (final FieldVector fv = Arrow.toArrowVector(alloc, this)) {
      return fv.toString();
    }
  }

  @Override
  public String toString() {
    try (final BufferAllocator alloc = new RootAllocator()) {
      return toString(alloc);
    }
  }

  public RowVector asRowVector() {
    if (this instanceof RowVector) {
      Preconditions.checkState(encoding == VectorEncoding.ROW);
      return (RowVector) this;
    }
    if (encoding == VectorEncoding.ROW) {
      throw new VeloxException(
          String.format(
              "The BaseVector has encoding ROW but was not wrapped"
                  + " as a Velox4J RowVector. Actual class: %s",
              getClass()));
    }
    throw new VeloxException(String.format("Not a RowVector. Encoding: %s", getEncoding()));
  }
}
