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
package com.facebook.velox4j.iterator;

import java.util.Iterator;

import com.facebook.velox4j.data.RowVector;

public final class DownIterators {
  /** From Java iterator */
  public static DownIterator fromJavaIterator(Iterator<RowVector> itr) {
    return new FromJavaIterator(itr);
  }

  private static class FromJavaIterator extends BaseDownIterator {
    private final Iterator<RowVector> itr;

    private FromJavaIterator(Iterator<RowVector> itr) {
      this.itr = itr;
    }

    @Override
    public State advanceInternal() {
      if (!itr.hasNext()) {
        return State.FINISHED;
      }
      return State.AVAILABLE;
    }

    @Override
    public void waitFor() throws InterruptedException {
      throw new IllegalStateException("#waitFor is called while the iterator doesn't block");
    }

    @Override
    public RowVector getInternal() {
      return itr.next();
    }

    @Override
    public void close() {}
  }

  private abstract static class BaseDownIterator implements DownIterator {
    protected BaseDownIterator() {}

    @Override
    public final int advance() {
      return advanceInternal().getId();
    }

    @Override
    public final long get() {
      return getInternal().id();
    }

    protected abstract DownIterator.State advanceInternal();

    protected abstract RowVector getInternal();
  }
}
