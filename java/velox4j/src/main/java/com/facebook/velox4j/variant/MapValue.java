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
package com.facebook.velox4j.variant;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;

public class MapValue extends Variant {
  private final Map<Variant, Variant> map;

  public MapValue(Map<Variant, Variant> map) {
    if (map == null) {
      this.map = null;
      return;
    }
    // The following is basically for test code, to write the map values into JSON with a
    //  comparatively stable order.
    // TODO: This may cause slow serialization as Variant#toString may be slow.
    //  A better way is to write reliable Variant#compareTo implementations for all variants.
    final TreeMap<Variant, Variant> builder =
        new TreeMap<>(Comparator.comparing(Variant::toString));
    builder.putAll(map);
    this.map = Collections.unmodifiableSortedMap(builder);
  }

  @JsonCreator
  private static MapValue create(@JsonProperty("value") Entries entries) {
    if (entries == null) {
      return new MapValue(null);
    }
    final int size = entries.size();
    final Map<Variant, Variant> builder = new HashMap<>(size);
    for (int i = 0; i < size; i++) {
      builder.put(entries.getKeys().get(i), entries.getValues().get(i));
    }
    Preconditions.checkArgument(
        builder.size() == size, "Duplicated keys found in entries while creating MapValue");
    return new MapValue(builder);
  }

  @JsonGetter("value")
  @JsonInclude(JsonInclude.Include.ALWAYS)
  public Entries getEntries() {
    if (map == null) {
      return null;
    }
    final int size = map.size();
    final List<Variant> keys = new ArrayList<>(size);
    final List<Variant> values = new ArrayList<>(size);
    for (Map.Entry<Variant, Variant> entry : map.entrySet()) {
      keys.add(entry.getKey());
      values.add(entry.getValue());
    }
    return new Entries(keys, values);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    MapValue mapValue = (MapValue) o;
    return Objects.equals(map, mapValue.map);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(map);
  }

  @Override
  public String toString() {
    return "MapValue{" + "map=" + map + '}';
  }

  public static class Entries {
    private final List<Variant> keys;
    private final List<Variant> values;

    @JsonCreator
    private Entries(
        @JsonProperty("keys") List<Variant> keys, @JsonProperty("values") List<Variant> values) {
      Preconditions.checkArgument(
          keys.size() == values.size(), "Entries should have same number of keys and values");
      Variants.checkSameType(keys);
      Variants.checkSameType(values);
      this.keys = Collections.unmodifiableList(keys);
      this.values = Collections.unmodifiableList(values);
    }

    @JsonGetter("keys")
    public List<Variant> getKeys() {
      return keys;
    }

    @JsonGetter("values")
    public List<Variant> getValues() {
      return values;
    }

    public int size() {
      return keys.size();
    }
  }
}
