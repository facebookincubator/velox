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
package com.facebook.velox4j.test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.apache.arrow.memory.RootAllocator;
import org.junit.Assert;

import com.facebook.velox4j.collection.Streams;
import com.facebook.velox4j.data.RowVector;
import com.facebook.velox4j.iterator.CloseableIterator;
import com.facebook.velox4j.iterator.UpIterator;
import com.facebook.velox4j.iterator.UpIterators;
import com.facebook.velox4j.serde.Serde;

public final class UpIteratorTests {
  public static RowVector collectSingleVector(UpIterator itr) {
    final List<RowVector> vectors = collect(itr);
    Assert.assertEquals(1, vectors.size());
    return vectors.get(0);
  }

  public static List<RowVector> collect(UpIterator itr) {
    final List<RowVector> vectors =
        Streams.fromIterator(UpIterators.asJavaIterator(itr)).collect(Collectors.toList());
    return vectors;
  }

  public static IteratorAssertionBuilder assertIterator(UpIterator itr) {
    return new IteratorAssertionBuilder(itr);
  }

  public static class IteratorAssertionBuilder {
    private final RootAllocator alloc = new RootAllocator();
    private final CloseableIterator<RowVector> itr;
    private final List<Consumer<Argument>> assertions = new ArrayList<>();
    private final List<Runnable> finalAssertions = new ArrayList<>();

    private IteratorAssertionBuilder(UpIterator itr) {
      this.itr = UpIterators.asJavaIterator(itr);
    }

    public IteratorAssertionBuilder assertNumRowVectors(int expected) {
      final AtomicInteger count = new AtomicInteger();
      assertForEach(
          new Consumer<Argument>() {
            @Override
            public void accept(Argument argument) {
              count.getAndIncrement();
            }
          });
      assertFinal(
          new Runnable() {
            @Override
            public void run() {
              Assert.assertEquals(expected, count.get());
            }
          });
      return this;
    }

    public IteratorAssertionBuilder assertRowVectorToString(int i, String expected) {
      return assertRowVector(
          i,
          new Consumer<RowVector>() {
            @Override
            public void accept(RowVector vector) {
              Assert.assertEquals(expected, vector.toString(alloc));
            }
          });
    }

    public IteratorAssertionBuilder assertRowVectorTypeJson(int i, String typeJsonExpected) {
      return assertRowVector(
          i,
          new Consumer<RowVector>() {
            @Override
            public void accept(RowVector vector) {
              Assert.assertEquals(typeJsonExpected, Serde.toPrettyJson(vector.getType()));
            }
          });
    }

    public IteratorAssertionBuilder assertRowVector(int i, Consumer<RowVector> body) {
      assertForEach(
          new Consumer<Argument>() {
            @Override
            public void accept(Argument argument) {
              if (argument.i == i) {
                body.accept(argument.rv);
              }
            }
          });
      return this;
    }

    private void assertForEach(Consumer<Argument> body) {
      assertions.add(body);
    }

    private void assertFinal(Runnable body) {
      finalAssertions.add(body);
    }

    public void run() {
      int i = 0;
      while (itr.hasNext()) {
        final RowVector rv = itr.next();
        for (Consumer<Argument> assertion : assertions) {
          assertion.accept(new Argument(i, rv));
        }
        i++;
      }
      for (Runnable r : finalAssertions) {
        r.run();
      }
      alloc.close();
    }

    private static class Argument {
      private final int i;
      private final RowVector rv;

      private Argument(int i, RowVector rv) {
        this.i = i;
        this.rv = rv;
      }
    }
  }
}
