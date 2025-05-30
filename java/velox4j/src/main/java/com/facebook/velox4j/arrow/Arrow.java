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
package com.facebook.velox4j.arrow;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.table.Table;

import com.facebook.velox4j.data.BaseVector;
import com.facebook.velox4j.data.RowVector;
import com.facebook.velox4j.jni.JniApi;
import com.facebook.velox4j.jni.StaticJniApi;

public class Arrow {
  private final JniApi jniApi;

  public Arrow(JniApi jniApi) {
    this.jniApi = jniApi;
  }

  public static FieldVector toArrowVector(BufferAllocator alloc, BaseVector vector) {
    try (final ArrowSchema cSchema = ArrowSchema.allocateNew(alloc);
        final ArrowArray cArray = ArrowArray.allocateNew(alloc)) {
      StaticJniApi.get().baseVectorToArrow(vector, cSchema, cArray);
      final FieldVector fv = Data.importVector(alloc, cArray, cSchema, null);
      return fv;
    }
  }

  public static Table toArrowTable(BufferAllocator alloc, RowVector vector) {
    try (final ArrowSchema cSchema = ArrowSchema.allocateNew(alloc);
        final ArrowArray cArray = ArrowArray.allocateNew(alloc)) {
      StaticJniApi.get().baseVectorToArrow(vector, cSchema, cArray);
      final VectorSchemaRoot vsr = Data.importVectorSchemaRoot(alloc, cArray, cSchema, null);
      return new Table(vsr);
    }
  }

  public BaseVector fromArrowVector(BufferAllocator alloc, FieldVector arrowVector) {
    try (final ArrowSchema cSchema = ArrowSchema.allocateNew(alloc);
        final ArrowArray cArray = ArrowArray.allocateNew(alloc)) {
      Data.exportVector(alloc, arrowVector, null, cArray, cSchema);
      final BaseVector imported = jniApi.arrowToBaseVector(cSchema, cArray);
      return imported;
    }
  }

  public RowVector fromArrowTable(BufferAllocator alloc, Table table) {
    try (final ArrowSchema cSchema = ArrowSchema.allocateNew(alloc);
        final ArrowArray cArray = ArrowArray.allocateNew(alloc);
        final VectorSchemaRoot vsr = table.toVectorSchemaRoot()) {
      Data.exportVectorSchemaRoot(alloc, vsr, null, cArray, cSchema);
      final BaseVector imported = jniApi.arrowToBaseVector(cSchema, cArray);
      return imported.asRowVector();
    }
  }
}
