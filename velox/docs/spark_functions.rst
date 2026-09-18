***********************
Spark Functions
***********************

The semantics of Spark functions align with
`Spark 3.5 <https://spark.apache.org/docs/3.5.0/api/sql/index.html>`_.
In the function descriptions, a function is marked as *ANSI compliant*
if it adheres to ANSI standard, subject to the :doc:`spark.ansi_enabled <configs>`
configuration. Otherwise, it simply follows Spark's semantics in ANSI OFF mode.

.. toctree::
    :maxdepth: 1

    functions/spark/math
    functions/spark/bitwise
    functions/spark/decimal
    functions/spark/comparison
    functions/spark/string
    functions/spark/datetime
    functions/spark/array
    functions/spark/map
    functions/spark/misc
    functions/spark/regexp
    functions/spark/binary
    functions/spark/aggregate
    functions/spark/window
    functions/spark/conversion
    functions/spark/url
    functions/spark/json

Here is a list of all scalar and aggregate Spark functions available in Velox.
Function names link to function descriptions. Check out coverage maps
for :doc:`all <functions/spark/coverage>` functions.

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

    ==================================================  ==================================================  ==================================================  ==  ==================================================  ==  ==================================================
    Scalar Functions                                                                                                                                                Aggregate Functions                                     Window Functions
    ==========================================================================================================================================================  ==  ==================================================  ==  ==================================================
    :spark:func:`abs`                                   :spark:func:`divide`                                :spark:func:`raise_error`                               :spark:func:`approx_percentile`                         :spark:func:`dense_rank`
    :spark:func:`acos`                                  :spark:func:`divide_deny_precision_loss`            :spark:func:`rand`                                      :spark:func:`avg`                                       :spark:func:`nth_value`
    :spark:func:`acosh`                                 :spark:func:`element_at`                            :spark:func:`randn`                                     :spark:func:`bit_xor`                                   :spark:func:`ntile`
    :spark:func:`add`                                   :spark:func:`empty2null`                            :spark:func:`random`                                    :spark:func:`bitmap_construct_agg`                      :spark:func:`rank`
    :spark:func:`add_deny_precision_loss`               :spark:func:`endswith`                              :spark:func:`randstr`                                   :spark:func:`bitmap_or_agg`                             :spark:func:`row_number`
    :spark:func:`add_months`                            :spark:func:`equalnullsafe`                         :spark:func:`read_side_padding`                         :spark:func:`bloom_filter_agg`
    :spark:func:`aggregate`                             :spark:func:`equalto`                               :spark:func:`regexp_extract`                            :spark:func:`collect_list`
    :spark:func:`array`                                 :spark:func:`exists`                                :spark:func:`regexp_extract_all`                        :spark:func:`collect_set`
    :spark:func:`array_append`                          :spark:func:`exp`                                   :spark:func:`regexp_instr`                              :spark:func:`corr`
    :spark:func:`array_compact`                         :spark:func:`expm1`                                 :spark:func:`regexp_replace`                            :spark:func:`covar_samp`
    :spark:func:`array_contains`                        :spark:func:`factorial`                             :spark:func:`remainder`                                 :spark:func:`first`
    :spark:func:`array_distinct`                        :spark:func:`filter`                                :spark:func:`repeat`                                    :spark:func:`first_ignore_null`
    :spark:func:`array_except`                          :spark:func:`find_in_set`                           :spark:func:`replace`                                   :spark:func:`kurtosis`
    :spark:func:`array_insert`                          :spark:func:`flatten`                               :spark:func:`reverse`                                   :spark:func:`last`
    :spark:func:`array_intersect`                       :spark:func:`floor`                                 :spark:func:`rint`                                      :spark:func:`last_ignore_null`
    :spark:func:`array_join`                            :spark:func:`forall`                                :spark:func:`rlike`                                     :spark:func:`max`
    :spark:func:`array_max`                             :spark:func:`format_number`                         :spark:func:`round`                                     :spark:func:`max_by`
    :spark:func:`array_min`                             :spark:func:`from_unixtime`                         :spark:func:`rpad`                                      :spark:func:`min`
    :spark:func:`array_position`                        :spark:func:`from_utc_timestamp`                    :spark:func:`rtrim`                                     :spark:func:`min_by`
    :spark:func:`array_prepend`                         :spark:func:`get`                                   :spark:func:`sec`                                       :spark:func:`mode`
    :spark:func:`array_remove`                          :spark:func:`get_json_object`                       :spark:func:`second`                                    :spark:func:`regr_replacement`
    :spark:func:`array_repeat`                          :spark:func:`get_timestamp`                         :spark:func:`sequence`                                  :spark:func:`skewness`
    :spark:func:`array_sort`                            :spark:func:`greaterthan`                           :spark:func:`sha1`                                      :spark:func:`stddev`
    :spark:func:`array_sort_desc`                       :spark:func:`greaterthanorequal`                    :spark:func:`sha2`                                      :spark:func:`stddev_samp`
    :spark:func:`array_union`                           :spark:func:`greatest`                              :spark:func:`shiftleft`                                 :spark:func:`sum`
    :spark:func:`arrays_zip`                            :spark:func:`hash`                                  :spark:func:`shiftright`                                :spark:func:`var_samp`
    :spark:func:`ascii`                                 :spark:func:`hash_with_seed`                        :spark:func:`shuffle`                                   :spark:func:`variance`
    :spark:func:`asin`                                  :spark:func:`hex`                                   :spark:func:`sign`
    :spark:func:`asinh`                                 :spark:func:`hour`                                  :spark:func:`sin`
    :spark:func:`assert_not_null`                       :spark:func:`hypot`                                 :spark:func:`sinh`
    :spark:func:`atan`                                  :spark:func:`initcap`                               :spark:func:`size`
    :spark:func:`atan2`                                 :spark:func:`instr`                                 :spark:func:`slice`
    :spark:func:`atanh`                                 :spark:func:`isnan`                                 :spark:func:`sort_array`
    :spark:func:`base64`                                :spark:func:`isnotnull`                             :spark:func:`soundex`
    :spark:func:`between`                               :spark:func:`isnull`                                :spark:func:`spark_partition_id`
    :spark:func:`bin`                                   :spark:func:`json_array_length`                     :spark:func:`split`
    :spark:func:`bit_count`                             :spark:func:`json_object_keys`                      :spark:func:`sqrt`
    :spark:func:`bit_get`                               :spark:func:`last_day`                              :spark:func:`startswith`
    :spark:func:`bit_length`                            :spark:func:`least`                                 :spark:func:`str_to_map`
    :spark:func:`bitwise_and`                           :spark:func:`left`                                  :spark:func:`substring`
    :spark:func:`bitwise_not`                           :spark:func:`length`                                :spark:func:`substring_index`
    :spark:func:`bitwise_or`                            :spark:func:`lessthan`                              :spark:func:`subtract`
    :spark:func:`bitwise_xor`                           :spark:func:`lessthanorequal`                       :spark:func:`subtract_deny_precision_loss`
    :spark:func:`cbrt`                                  :spark:func:`levenshtein`                           :spark:func:`tan`
    :spark:func:`ceil`                                  :spark:func:`like`                                  :spark:func:`tanh`
    :spark:func:`char_type_write_side_check`            :spark:func:`ln`                                    :spark:func:`timestamp_micros`
    :spark:func:`checked_add`                           :spark:func:`locate`                                :spark:func:`timestamp_millis`
    :spark:func:`checked_add_deny_precision_loss`       :spark:func:`log`                                   :spark:func:`timestamp_seconds`
    :spark:func:`checked_div`                           :spark:func:`log10`                                 :spark:func:`timestampadd`
    :spark:func:`checked_divide`                        :spark:func:`log1p`                                 :spark:func:`timestampdiff`
    :spark:func:`checked_multiply`                      :spark:func:`log2`                                  :spark:func:`to_json`
    :spark:func:`checked_multiply_deny_precision_loss`  :spark:func:`lower`                                 :spark:func:`to_pretty_string`
    :spark:func:`checked_subtract`                      :spark:func:`lpad`                                  :spark:func:`to_unix_timestamp`
    :spark:func:`checked_subtract_deny_precision_loss`  :spark:func:`ltrim`                                 :spark:func:`to_utc_timestamp`
    :spark:func:`chr`                                   :spark:func:`luhn_check`                            :spark:func:`transform`
    :spark:func:`concat`                                :spark:func:`make_date`                             :spark:func:`transform_values`
    :spark:func:`contains`                              :spark:func:`make_timestamp`                        :spark:func:`translate`
    :spark:func:`conv`                                  :spark:func:`make_ym_interval`                      :spark:func:`trim`
    :spark:func:`cos`                                   :spark:func:`map`                                   :spark:func:`trunc`
    :spark:func:`cosh`                                  :spark:func:`map_concat`                            :spark:func:`unaryminus`
    :spark:func:`cot`                                   :spark:func:`map_entries`                           :spark:func:`unbase64`
    :spark:func:`crc32`                                 :spark:func:`map_filter`                            :spark:func:`unhex`
    :spark:func:`csc`                                   :spark:func:`map_from_arrays`                       :spark:func:`unix_date`
    :spark:func:`date_add`                              :spark:func:`map_from_entries`                      :spark:func:`unix_micros`
    :spark:func:`date_format`                           :spark:func:`map_keys`                              :spark:func:`unix_millis`
    :spark:func:`date_from_unix_date`                   :spark:func:`map_values`                            :spark:func:`unix_seconds`
    :spark:func:`date_sub`                              :spark:func:`map_zip_with`                          :spark:func:`unix_timestamp`
    :spark:func:`date_trunc`                            :spark:func:`mask`                                  :spark:func:`unscaled_value`
    :spark:func:`datediff`                              :spark:func:`md5`                                   :spark:func:`upper`
    :spark:func:`day`                                   :spark:func:`might_contain`                         :spark:func:`url_decode`
    :spark:func:`dayname`                               :spark:func:`minute`                                :spark:func:`url_encode`
    :spark:func:`dayofmonth`                            :spark:func:`monotonically_increasing_id`           :spark:func:`uuid`
    :spark:func:`dayofweek`                             :spark:func:`month`                                 :spark:func:`varchar_type_write_side_check`
    :spark:func:`dayofyear`                             :spark:func:`monthname`                             :spark:func:`week_of_year`
    :spark:func:`decimal_equalto`                       :spark:func:`months_between`                        :spark:func:`weekday`
    :spark:func:`decimal_greaterthan`                   :spark:func:`multiply`                              :spark:func:`width_bucket`
    :spark:func:`decimal_greaterthanorequal`            :spark:func:`multiply_deny_precision_loss`          :spark:func:`xxhash64`
    :spark:func:`decimal_lessthan`                      :spark:func:`next_day`                              :spark:func:`xxhash64_with_seed`
    :spark:func:`decimal_lessthanorequal`               :spark:func:`overlay`                               :spark:func:`year`
    :spark:func:`decimal_notequalto`                    :spark:func:`pmod`                                  :spark:func:`year_of_week`
    :spark:func:`decode`                                :spark:func:`power`                                 :spark:func:`zip_with`
    :spark:func:`degrees`                               :spark:func:`quarter`
    :spark:func:`div`                                   :spark:func:`radians`
    ==================================================  ==================================================  ==================================================  ==  ==================================================  ==  ==================================================
