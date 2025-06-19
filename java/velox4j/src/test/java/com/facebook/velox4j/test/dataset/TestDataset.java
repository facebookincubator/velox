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
package com.facebook.velox4j.test.dataset;

import java.util.List;

/**
 * An interface to create dataset for testing purpose. The generic type `FID` stands for the ID type
 * of all files in the dataset.
 */
public interface TestDataset<FID> {
  /** Lists all files in the dataset. */
  List<FID> listFiles();

  /** Returns one single test data file by the input file ID. */
  TestDataFile get(FID fid);
}
