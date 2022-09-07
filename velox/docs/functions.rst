***********************
Presto Functions
***********************

.. toctree::
    :maxdepth: 1

    functions/math
    functions/bitwise
    functions/comparison
    functions/string
    functions/datetime
    functions/array
    functions/map
    functions/regexp
    functions/binary
    functions/json
    functions/url
    functions/aggregate
    functions/window
    functions/hyperloglog

Here is a list of all scalar and aggregate Presto functions available in Velox.
Function names link to function descriptions. Check out coverage maps
for :doc:`all <functions/coverage>` and :doc:`most used
<functions/most_used_coverage>` functions for broader context.

.. raw:: html

    <style>

    table.rows th {
        background-color: lightblue;
        border-style: solid solid solid solid;
        border-width: 1px 1px 1px 1px;
        border-color: #AAAAAA;
        text-align: center;
    }

    table.rows td {
        border-style: solid solid solid solid;
        border-width: 1px 1px 1px 1px;
        border-color: #AAAAAA;
    }

    table.rows tr {
        border-style: solid solid solid solid;
        border-width: 0px 0px 0px 0px;
        border-color: #AAAAAA;
    }

    table.rows td:nth-child(4) {
        background-color: lightblue;
    }
    </style>

.. table::
    :widths: auto
    :class: rows

    ======================================  ======================================  ======================================  ==  ======================================
    Scalar Functions                                                                                                            Aggregate Functions
    ======================================================================================================================  ==  ======================================
    :func:`abs`                             :func:`element_at`                      :func:`pow`                                 :func:`approx_distinct`
    :func:`acos`                            :func:`empty_approx_set`                :func:`power`                               :func:`approx_most_frequent`
    :func:`array_constructor`               :func:`eq`                              :func:`quarter`                             :func:`approx_percentile`
    :func:`array_distinct`                  :func:`exp`                             :func:`radians`                             :func:`approx_set`
    :func:`array_duplicates`                :func:`filter`                          :func:`rand`                                :func:`arbitrary`
    :func:`array_except`                    :func:`floor`                           :func:`random`                              :func:`array_agg`
    :func:`array_intersect`                 :func:`format_datetime`                 :func:`reduce`                              :func:`avg`
    :func:`array_join`                      :func:`from_base`                       :func:`regexp_extract`                      :func:`bitwise_and_agg`
    :func:`array_max`                       :func:`from_base64`                     :func:`regexp_extract_all`                  :func:`bitwise_or_agg`
    :func:`array_min`                       :func:`from_hex`                        :func:`regexp_like`                         :func:`bool_and`
    :func:`array_position`                  :func:`from_unixtime`                   :func:`regexp_replace`                      :func:`bool_or`
    :func:`array_sort`                      :func:`greatest`                        :func:`replace`                             :func:`checksum`
    :func:`array_sum`                       :func:`gt`                              :func:`reverse`                             :func:`corr`
    :func:`array_sum_alt`                   :func:`gte`                             :func:`round`                               :func:`count`
    :func:`arrays_overlap`                  :func:`hour`                            :func:`rpad`                                :func:`count_if`
    :func:`asin`                            in                                      :func:`rtrim`                               :func:`covar_pop`
    :func:`atan`                            :func:`infinity`                        :func:`second`                              :func:`covar_samp`
    :func:`atan2`                           :func:`is_finite`                       :func:`sha256`                              :func:`every`
    :func:`between`                         :func:`is_infinite`                     :func:`sha512`                              :func:`histogram`
    :func:`bit_count`                       :func:`is_json_scalar`                  :func:`sign`                                :func:`map_agg`
    :func:`bitwise_and`                     :func:`is_nan`                          :func:`sin`                                 :func:`map_union`
    :func:`bitwise_arithmetic_shift_right`  :func:`is_null`                         :func:`slice`                               :func:`max`
    :func:`bitwise_left_shift`              :func:`json_array_contains`             :func:`split`                               :func:`max_by`
    :func:`bitwise_logical_shift_right`     :func:`json_array_length`               :func:`split_part`                          :func:`max_data_size_for_stats`
    :func:`bitwise_not`                     :func:`json_extract_scalar`             :func:`sqrt`                                :func:`merge`
    :func:`bitwise_or`                      :func:`least`                           :func:`strpos`                              :func:`min`
    :func:`bitwise_right_shift`             :func:`length`                          :func:`subscript`                           :func:`min_by`
    :func:`bitwise_right_shift_arithmetic`  :func:`like`                            :func:`substr`                              :func:`stddev`
    :func:`bitwise_shift_left`              :func:`ln`                              :func:`tan`                                 :func:`stddev_pop`
    :func:`bitwise_xor`                     :func:`log10`                           :func:`tanh`                                :func:`stddev_samp`
    :func:`cardinality`                     :func:`log2`                            :func:`to_base`                             :func:`sum`
    :func:`cbrt`                            :func:`lower`                           :func:`to_base64`                           :func:`var_pop`
    :func:`ceil`                            :func:`lpad`                            :func:`to_hex`                              :func:`var_samp`
    :func:`ceiling`                         :func:`lt`                              :func:`to_unixtime`                         :func:`variance`
    :func:`chr`                             :func:`lte`                             :func:`to_utf8`
    :func:`clamp`                           :func:`ltrim`                           :func:`transform`
    :func:`codepoint`                       :func:`map`                             :func:`transform_keys`
    :func:`combinations`                    :func:`map_concat`                      :func:`transform_values`
    :func:`concat`                          :func:`map_concat_empty_nulls`          :func:`trim`
    :func:`contains`                        :func:`map_entries`                     :func:`upper`
    :func:`cos`                             :func:`map_filter`                      :func:`url_decode`
    :func:`cosh`                            :func:`map_keys`                        :func:`url_encode`
    :func:`date_add`                        :func:`map_values`                      :func:`url_extract_fragment`
    :func:`date_diff`                       :func:`md5`                             :func:`url_extract_host`
    :func:`date_format`                     :func:`millisecond`                     :func:`url_extract_parameter`
    :func:`date_parse`                      :func:`minus`                           :func:`url_extract_path`
    :func:`date_trunc`                      :func:`minute`                          :func:`url_extract_port`
    :func:`day`                             :func:`mod`                             :func:`url_extract_protocol`
    :func:`day_of_month`                    :func:`month`                           :func:`url_extract_query`
    :func:`day_of_week`                     :func:`multiply`                        :func:`width_bucket`
    :func:`day_of_year`                     :func:`nan`                             :func:`xxhash64`
    :func:`degrees`                         :func:`negate`                          :func:`year`
    :func:`distinct_from`                   :func:`neq`                             :func:`year_of_week`
    :func:`divide`                          not                                     :func:`yow`
    :func:`dow`                             :func:`parse_datetime`                  :func:`zip`
    :func:`doy`                             :func:`pi`
    :func:`e`                               :func:`plus`
    ======================================  ======================================  ======================================  ==  ======================================
