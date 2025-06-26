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
package com.facebook.velox4j.session;

import com.facebook.velox4j.arrow.Arrow;
import com.facebook.velox4j.connector.ExternalStreams;
import com.facebook.velox4j.data.BaseVectors;
import com.facebook.velox4j.data.SelectivityVectors;
import com.facebook.velox4j.eval.Evaluations;
import com.facebook.velox4j.jni.CppObject;
import com.facebook.velox4j.plan.TableWriteNode;
import com.facebook.velox4j.query.Queries;
import com.facebook.velox4j.serializable.ISerializables;
import com.facebook.velox4j.variant.Variants;
import com.facebook.velox4j.write.TableWriteTraits;

/**
 * A Velox4J session consists of a set of active Velox4J APIs.
 *
 * <p>Session itself should be closed after use, as it's a CppObject. Once it is closed, all the
 * created C++ objects will be destroyed to avoid memory leakage.
 */
public interface Session extends CppObject {
  /** APIs in relation to {@link com.facebook.velox4j.eval.Evaluation}. */
  Evaluations evaluationOperations();

  /** APIs in relation to {@link com.facebook.velox4j.query.Query}. */
  Queries queryOperations();

  /** APIs in relation to {@link com.facebook.velox4j.connector.ExternalStream}. */
  ExternalStreams externalStreamOperations();

  /** APIs in relation to {@link com.facebook.velox4j.data.BaseVector}. */
  BaseVectors baseVectorOperations();

  /** APIs in relation to {@link com.facebook.velox4j.data.SelectivityVector}. */
  SelectivityVectors selectivityVectorOperations();

  /**
   * Arrow APIs for vectors. This includes interchange functionalities between Velox native vector
   * format and Arrow-Java format.
   */
  Arrow arrowOperations();

  /**
   * An API for creating certain required information for building a {@link TableWriteNode} in Java.
   */
  TableWriteTraits tableWriteTraitsOperations();

  /** APIs in relation to {@link com.facebook.velox4j.serializable.ISerializable}. */
  ISerializables iSerializableOperations();

  /** APIs in relation to {@link com.facebook.velox4j.variant.Variant}. */
  Variants variantOperations();
}
