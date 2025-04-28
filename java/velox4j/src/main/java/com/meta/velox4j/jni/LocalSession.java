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

import com.meta.velox4j.arrow.Arrow;
import com.meta.velox4j.connector.ExternalStreams;
import com.meta.velox4j.data.BaseVectors;
import com.meta.velox4j.data.RowVectors;
import com.meta.velox4j.data.SelectivityVectors;
import com.meta.velox4j.eval.Evaluations;
import com.meta.velox4j.query.Queries;
import com.meta.velox4j.serializable.ISerializables;
import com.meta.velox4j.session.Session;
import com.meta.velox4j.variant.Variants;
import com.meta.velox4j.write.TableWriteTraits;

public class LocalSession implements Session {
  private final long id;

  LocalSession(long id) {
    this.id = id;
  }

  private JniApi jniApi() {
    final JniWrapper jniWrapper = new JniWrapper(this.id);
    return new JniApi(jniWrapper);
  }

  @Override
  public long id() {
    return id;
  }

  @Override
  public Evaluations evaluationOps() {
    return new Evaluations(jniApi());
  }

  @Override
  public Queries queryOps() {
    return new Queries(jniApi());
  }

  @Override
  public ExternalStreams externalStreamOps() {
    return new ExternalStreams(jniApi());
  }

  @Override
  public BaseVectors baseVectorOps() {
    return new BaseVectors(jniApi());
  }

  @Override
  public RowVectors rowVectorOps() {
    return new RowVectors(jniApi());
  }

  @Override
  public SelectivityVectors selectivityVectorOps() {
    return new SelectivityVectors(jniApi());
  }

  @Override
  public Arrow arrowOps() {
    return new Arrow(jniApi());
  }

  @Override
  public TableWriteTraits tableWriteTraitsOps() {
    return new TableWriteTraits(jniApi());
  }

  @Override
  public ISerializables iSerializableOps() {
    return new ISerializables(jniApi());
  }

  @Override
  public Variants variantOps() {
    return new Variants(jniApi());
  }
}
