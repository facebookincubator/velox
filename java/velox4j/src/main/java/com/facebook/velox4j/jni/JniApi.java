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

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import com.google.common.annotations.VisibleForTesting;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;

import com.facebook.velox4j.connector.ExternalStream;
import com.facebook.velox4j.connector.ExternalStreams;
import com.facebook.velox4j.data.BaseVector;
import com.facebook.velox4j.data.RowVector;
import com.facebook.velox4j.data.SelectivityVector;
import com.facebook.velox4j.data.VectorEncoding;
import com.facebook.velox4j.eval.Evaluation;
import com.facebook.velox4j.eval.Evaluator;
import com.facebook.velox4j.iterator.DownIterator;
import com.facebook.velox4j.iterator.GenericUpIterator;
import com.facebook.velox4j.iterator.UpIterator;
import com.facebook.velox4j.plan.AggregationNode;
import com.facebook.velox4j.query.Query;
import com.facebook.velox4j.query.QueryExecutor;
import com.facebook.velox4j.query.SerialTask;
import com.facebook.velox4j.serde.Serde;
import com.facebook.velox4j.serializable.ISerializable;
import com.facebook.velox4j.serializable.ISerializableCpp;
import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.type.Type;
import com.facebook.velox4j.variant.Variant;
import com.facebook.velox4j.variant.VariantCpp;

/**
 * The higher-level JNI-based API over {@link JniWrapper}. The API hides details like native
 * pointers and serialized data from developers, instead provides objective forms of the required
 * functionalities.
 */
public final class JniApi {
  private final JniWrapper jni;

  JniApi(JniWrapper jni) {
    this.jni = jni;
  }

  public Evaluator createEvaluator(Evaluation evaluation) {
    final String evalJson = Serde.toPrettyJson(evaluation);
    return new Evaluator(this, jni.createEvaluator(evalJson));
  }

  public BaseVector evaluatorEval(Evaluator evaluator, SelectivityVector sv, RowVector input) {
    return baseVectorWrap(jni.evaluatorEval(evaluator.id(), sv.id(), input.id()));
  }

  public QueryExecutor createQueryExecutor(Query query) {
    final String queryJson = Serde.toPrettyJson(query);
    return new QueryExecutor(this, jni.createQueryExecutor(queryJson));
  }

  @VisibleForTesting
  QueryExecutor createQueryExecutor(String queryJson) {
    return new QueryExecutor(this, jni.createQueryExecutor(queryJson));
  }

  public SerialTask queryExecutorExecute(QueryExecutor executor) {
    return new SerialTask(this, jni.queryExecutorExecute(executor.id()));
  }

  public RowVector upIteratorGet(UpIterator itr) {
    return baseVectorWrap(jni.upIteratorGet(itr.id())).asRowVector();
  }

  public ExternalStream createExternalStreamFromDownIterator(DownIterator itr) {
    return new ExternalStreams.GenericExternalStream(jni.createExternalStreamFromDownIterator(itr));
  }

  public ExternalStreams.BlockingQueue createBlockingQueue() {
    return new ExternalStreams.BlockingQueue(jni.createBlockingQueue());
  }

  public BaseVector createEmptyBaseVector(Type type) {
    final String typeJson = Serde.toJson(type);
    return baseVectorWrap(jni.createEmptyBaseVector(typeJson));
  }

  public BaseVector arrowToBaseVector(ArrowSchema schema, ArrowArray array) {
    return baseVectorWrap(jni.arrowToBaseVector(schema.memoryAddress(), array.memoryAddress()));
  }

  public List<BaseVector> baseVectorDeserialize(String serialized) {
    return Arrays.stream(jni.baseVectorDeserialize(serialized))
        .mapToObj(this::baseVectorWrap)
        .collect(Collectors.toList());
  }

  public BaseVector baseVectorWrapInConstant(BaseVector vector, int length, int index) {
    return baseVectorWrap(jni.baseVectorWrapInConstant(vector.id(), length, index));
  }

  public BaseVector baseVectorSlice(BaseVector vector, int offset, int length) {
    return baseVectorWrap(jni.baseVectorSlice(vector.id(), offset, length));
  }

  public BaseVector loadedVector(BaseVector vector) {
    return baseVectorWrap(jni.baseVectorLoadedVector(vector.id()));
  }

  public SelectivityVector createSelectivityVector(int length) {
    return new SelectivityVector(jni.createSelectivityVector(length));
  }

  public RowType tableWriteTraitsOutputTypeWithAggregationNode(AggregationNode aggregationNode) {
    final String aggregationNodeJson = Serde.toJson(aggregationNode);
    final String typeJson = jni.tableWriteTraitsOutputTypeWithAggregationNode(aggregationNodeJson);
    final RowType type = Serde.fromJson(typeJson, RowType.class);
    return type;
  }

  public ISerializableCpp iSerializableAsCpp(ISerializable iSerializable) {
    final String json = Serde.toPrettyJson(iSerializable);
    return new ISerializableCpp(jni.iSerializableAsCpp(json));
  }

  public VariantCpp variantAsCpp(Variant variant) {
    final String json = Serde.toPrettyJson(variant);
    return new VariantCpp(jni.variantAsCpp(json));
  }

  @VisibleForTesting
  public UpIterator createUpIteratorWithExternalStream(ExternalStream es) {
    return new GenericUpIterator(this, jni.createUpIteratorWithExternalStream(es.id()));
  }

  private BaseVector baseVectorWrap(long id) {
    final VectorEncoding encoding =
        VectorEncoding.valueOf(StaticJniWrapper.get().baseVectorGetEncoding(id));
    return BaseVector.wrap(this, id, encoding);
  }
}
