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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.table.Table;

import com.facebook.velox4j.arrow.Arrow;
import com.facebook.velox4j.jni.JniApi;

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
