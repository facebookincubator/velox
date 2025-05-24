package com.meta.velox4j.data;

import com.google.common.base.Preconditions;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;

import com.meta.velox4j.arrow.Arrow;
import com.meta.velox4j.exception.VeloxException;
import com.meta.velox4j.jni.CppObject;
import com.meta.velox4j.jni.JniApi;
import com.meta.velox4j.jni.StaticJniApi;
import com.meta.velox4j.type.Type;

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
