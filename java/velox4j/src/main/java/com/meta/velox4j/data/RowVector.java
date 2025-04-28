package com.meta.velox4j.data;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.table.Table;

import com.meta.velox4j.arrow.Arrow;
import com.meta.velox4j.jni.JniApi;

public class RowVector extends BaseVector {
  protected RowVector(JniApi jniApi, long id) {
    super(jniApi, id, VectorEncoding.ROW);
  }

  @Override
  public String toString(BufferAllocator alloc) {
    try (final Table t = Arrow.toArrowTable(alloc, this);
        final VectorSchemaRoot vsr = t.toVectorSchemaRoot()) {
      return vsr.contentToTSVString();
    }
  }
}
