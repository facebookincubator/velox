***********************
Presto Functions
***********************

.. toctree::
    :maxdepth: 1

    functions/presto/math
    functions/presto/decimal
    functions/presto/bitwise
    functions/presto/comparison
    functions/presto/string
    functions/presto/datetime
    functions/presto/array
    functions/presto/map
    functions/presto/regexp
    functions/presto/binary
    functions/presto/json
    functions/presto/conversion
    functions/presto/url
    functions/presto/aggregate
    functions/presto/window
    functions/presto/hyperloglog
    functions/presto/khyperloglog
    functions/presto/setdigest
    functions/presto/tdigest
    functions/presto/qdigest
    functions/presto/geospatial
    functions/presto/ipaddress
    functions/presto/uuid
    functions/presto/enum
    functions/presto/misc

Here is a list of all scalar and aggregate Presto functions available in Velox.
Function names link to function descriptions. Check out coverage maps
for :doc:`all <functions/presto/coverage>` and :doc:`most used
<functions/presto/most_used_coverage>` functions for broader context.

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

    table.rows td:nth-child(4), td:nth-child(6) {
        background-color: lightblue;
    }
    </style>

.. table::
    :widths: auto
    :class: rows

    =================================================  =================================================  =================================================  ==  =================================================  ==  =================================================
    Scalar Functions                                                                                                                                             Aggregate Functions                                    Window Functions
    =======================================================================================================================================================  ==  =================================================  ==  =================================================
    :func:`$internal$json_string_to_array_cast`        :func:`geometry_to_dissolved_bing_tiles`           :func:`secure_rand`                                    :func:`any_value`                                      :func:`cume_dist`
    :func:`$internal$json_string_to_map_cast`          :func:`geometry_union`                             :func:`secure_random`                                  :func:`approx_distinct`                                :func:`dense_rank`
    :func:`$internal$json_string_to_row_cast`          :func:`great_circle_distance`                      :func:`sequence`                                       :func:`approx_most_frequent`                           :func:`first_value`
    :func:`$internal$split_to_map`                     :func:`greatest`                                   :func:`sha1`                                           :func:`approx_percentile`                              :func:`lag`
    :func:`abs`                                        :func:`gt`                                         :func:`sha256`                                         :func:`approx_set`                                     :func:`last_value`
    :func:`acos`                                       :func:`gte`                                        :func:`sha512`                                         :func:`arbitrary`                                      :func:`lead`
    :func:`all_keys_match`                             :func:`hamming_distance`                           :func:`shuffle`                                        :func:`array_agg`                                      :func:`nth_value`
    :func:`all_match`                                  :func:`hmac_md5`                                   :func:`sign`                                           :func:`avg`                                            :func:`ntile`
    :func:`any_keys_match`                             :func:`hmac_sha1`                                  :func:`simplify_geometry`                              :func:`bitwise_and_agg`                                :func:`percent_rank`
    :func:`any_match`                                  :func:`hmac_sha256`                                :func:`sin`                                            :func:`bitwise_or_agg`                                 :func:`rank`
    :func:`any_values_match`                           :func:`hmac_sha512`                                :func:`slice`                                          :func:`bitwise_xor_agg`                                :func:`row_number`
    :func:`array_average`                              :func:`hour`                                       :func:`split`                                          :func:`bool_and`
    :func:`array_constructor`                          in                                                 :func:`split_part`                                     :func:`bool_or`
    :func:`array_cum_sum`                              :func:`infinity`                                   :func:`split_to_map`                                   :func:`checksum`
    :func:`array_distinct`                             :func:`inverse_beta_cdf`                           :func:`split_to_multimap`                              :func:`classification_fall_out`
    :func:`array_duplicates`                           :func:`inverse_binomial_cdf`                       :func:`spooky_hash_v2_32`                              :func:`classification_miss_rate`
    :func:`array_except`                               :func:`inverse_cauchy_cdf`                         :func:`spooky_hash_v2_64`                              :func:`classification_precision`
    :func:`array_frequency`                            :func:`inverse_chi_squared_cdf`                    :func:`sqrt`                                           :func:`classification_recall`
    :func:`array_has_duplicates`                       :func:`inverse_f_cdf`                              :func:`st_area`                                        :func:`classification_thresholds`
    :func:`array_intersect`                            :func:`inverse_gamma_cdf`                          :func:`st_asbinary`                                    :func:`corr`
    :func:`array_join`                                 :func:`inverse_laplace_cdf`                        :func:`st_astext`                                      :func:`count`
    :func:`array_max`                                  :func:`inverse_normal_cdf`                         :func:`st_boundary`                                    :func:`count_if`
    :func:`array_max_by`                               :func:`inverse_poisson_cdf`                        :func:`st_buffer`                                      :func:`covar_pop`
    :func:`array_min`                                  :func:`inverse_t_cdf`                              :func:`st_centroid`                                    :func:`covar_samp`
    :func:`array_min_by`                               :func:`inverse_weibull_cdf`                        :func:`st_contains`                                    :func:`entropy`
    :func:`array_normalize`                            :func:`ip_prefix`                                  :func:`st_convexhull`                                  :func:`every`
    :func:`array_position`                             :func:`ip_prefix_collapse`                         :func:`st_coorddim`                                    :func:`geometric_mean`
    :func:`array_remove`                               :func:`ip_prefix_subnets`                          :func:`st_crosses`                                     :func:`histogram`
    :func:`array_sort`                                 :func:`ip_subnet_max`                              :func:`st_difference`                                  :func:`kurtosis`
    :func:`array_sort_desc`                            :func:`ip_subnet_min`                              :func:`st_dimension`                                   :func:`map_agg`
    :func:`array_subset`                               :func:`ip_subnet_range`                            :func:`st_disjoint`                                    :func:`map_union`
    :func:`array_sum`                                  :func:`is_finite`                                  :func:`st_distance`                                    :func:`map_union_sum`
    :func:`array_sum_propagate_element_null`           :func:`is_infinite`                                :func:`st_endpoint`                                    :func:`max`
    :func:`array_top_n`                                :func:`is_json_scalar`                             :func:`st_envelope`                                    :func:`max_by`
    :func:`array_union`                                :func:`is_nan`                                     :func:`st_envelopeaspts`                               :func:`max_data_size_for_stats`
    :func:`arrays_overlap`                             :func:`is_null`                                    :func:`st_equals`                                      :func:`merge`
    :func:`asin`                                       :func:`is_private_ip`                              :func:`st_exteriorring`                                :func:`min`
    :func:`at_timezone`                                :func:`is_subnet_of`                               :func:`st_geometries`                                  :func:`min_by`
    :func:`atan`                                       :func:`json_array_contains`                        :func:`st_geometryfromtext`                            :func:`multimap_agg`
    :func:`atan2`                                      :func:`json_array_get`                             :func:`st_geometryn`                                   :func:`noisy_approx_distinct_sfm`
    :func:`beta_cdf`                                   :func:`json_array_length`                          :func:`st_geometrytype`                                :func:`noisy_approx_set_sfm`
    :func:`between`                                    :func:`json_extract`                               :func:`st_geomfrombinary`                              :func:`noisy_approx_set_sfm_from_index_and_zeros`
    :func:`bing_tile`                                  :func:`json_extract_scalar`                        :func:`st_interiorringn`                               :func:`noisy_avg_gaussian`
    :func:`bing_tile_at`                               :func:`json_format`                                :func:`st_interiorrings`                               :func:`noisy_count_gaussian`
    :func:`bing_tile_children`                         :func:`json_parse`                                 :func:`st_intersection`                                :func:`noisy_count_if_gaussian`
    :func:`bing_tile_coordinates`                      :func:`json_size`                                  :func:`st_intersects`                                  :func:`noisy_sum_gaussian`
    :func:`bing_tile_parent`                           :func:`laplace_cdf`                                :func:`st_isclosed`                                    :func:`numeric_histogram`
    :func:`bing_tile_polygon`                          :func:`last_day_of_month`                          :func:`st_isempty`                                     :func:`qdigest_agg`
    :func:`bing_tile_quadkey`                          :func:`least`                                      :func:`st_isring`                                      :func:`reduce_agg`
    :func:`bing_tile_zoom_level`                       :func:`length`                                     :func:`st_issimple`                                    :func:`regr_avgx`
    :func:`bing_tiles_around`                          :func:`levenshtein_distance`                       :func:`st_isvalid`                                     :func:`regr_avgy`
    :func:`binomial_cdf`                               :func:`like`                                       :func:`st_length`                                      :func:`regr_count`
    :func:`bit_count`                                  :func:`line_interpolate_point`                     :func:`st_linefromtext`                                :func:`regr_intercept`
    :func:`bit_length`                                 :func:`line_locate_point`                          :func:`st_linestring`                                  :func:`regr_r2`
    :func:`bitwise_and`                                :func:`ln`                                         :func:`st_multipoint`                                  :func:`regr_slope`
    :func:`bitwise_arithmetic_shift_right`             :func:`localtime`                                  :func:`st_numgeometries`                               :func:`regr_sxx`
    :func:`bitwise_left_shift`                         :func:`log10`                                      :func:`st_numinteriorring`                             :func:`regr_sxy`
    :func:`bitwise_logical_shift_right`                :func:`log2`                                       :func:`st_numpoints`                                   :func:`regr_syy`
    :func:`bitwise_not`                                :func:`longest_common_prefix`                      :func:`st_overlaps`                                    :func:`set_agg`
    :func:`bitwise_or`                                 :func:`lower`                                      :func:`st_point`                                       :func:`set_union`
    :func:`bitwise_right_shift`                        :func:`lpad`                                       :func:`st_pointn`                                      :func:`skewness`
    :func:`bitwise_right_shift_arithmetic`             :func:`lt`                                         :func:`st_points`                                      :func:`stddev`
    :func:`bitwise_shift_left`                         :func:`lte`                                        :func:`st_polygon`                                     :func:`stddev_pop`
    :func:`bitwise_xor`                                :func:`ltrim`                                      :func:`st_relate`                                      :func:`stddev_samp`
    :func:`cardinality`                                :func:`map`                                        :func:`st_startpoint`                                  :func:`sum`
    :func:`cauchy_cdf`                                 :func:`map_concat`                                 :func:`st_symdifference`                               :func:`sum_data_size_for_stats`
    :func:`cbrt`                                       :func:`map_entries`                                :func:`st_touches`                                     :func:`tdigest_agg`
    :func:`ceil`                                       :func:`map_filter`                                 :func:`st_union`                                       :func:`var_pop`
    :func:`ceiling`                                    :func:`map_from_entries`                           :func:`st_within`                                      :func:`var_samp`
    :func:`chi_squared_cdf`                            :func:`map_intersect`                              :func:`st_x`                                           :func:`variance`
    :func:`chr`                                        :func:`map_key_exists`                             :func:`st_xmax`
    :func:`clamp`                                      :func:`map_keys`                                   :func:`st_xmin`
    :func:`codepoint`                                  :func:`map_keys_by_top_n_values`                   :func:`st_y`
    :func:`combinations`                               :func:`map_normalize`                              :func:`st_ymax`
    :func:`combine_hash_internal`                      :func:`map_remove_null_values`                     :func:`st_ymin`
    :func:`concat`                                     :func:`map_subset`                                 :func:`starts_with`
    :func:`construct_tdigest`                          :func:`map_top_n`                                  :func:`strpos`
    :func:`contains`                                   :func:`map_top_n_keys`                             :func:`strrpos`
    :func:`cos`                                        :func:`map_top_n_values`                           :func:`subscript`
    :func:`cosh`                                       :func:`map_values`                                 :func:`substr`
    :func:`cosine_similarity`                          :func:`map_zip_with`                               :func:`substring`
    :func:`crc32`                                      :func:`md5`                                        :func:`t_cdf`
    :func:`current_date`                               :func:`merge_hll`                                  :func:`tan`
    :func:`date`                                       :func:`merge_sfm`                                  :func:`tanh`
    :func:`date_add`                                   :func:`merge_tdigest`                              :func:`timezone_hour`
    :func:`date_diff`                                  :func:`millisecond`                                :func:`timezone_minute`
    :func:`date_format`                                :func:`minus`                                      :func:`to_base`
    :func:`date_parse`                                 :func:`minute`                                     :func:`to_base64`
    :func:`date_trunc`                                 :func:`mod`                                        :func:`to_base64url`
    :func:`day`                                        :func:`month`                                      :func:`to_big_endian_32`
    :func:`day_of_month`                               :func:`multimap_from_entries`                      :func:`to_big_endian_64`
    :func:`day_of_week`                                :func:`multiply`                                   :func:`to_hex`
    :func:`day_of_year`                                :func:`murmur3_x64_128`                            :func:`to_ieee754_32`
    :func:`degrees`                                    :func:`nan`                                        :func:`to_ieee754_64`
    :func:`destructure_tdigest`                        :func:`negate`                                     :func:`to_iso8601`
    :func:`distinct_from`                              :func:`neq`                                        :func:`to_milliseconds`
    :func:`divide`                                     :func:`ngrams`                                     :func:`to_unixtime`
    :func:`dot_product`                                :func:`no_keys_match`                              :func:`to_utf8`
    :func:`dow`                                        :func:`no_values_match`                            :func:`trail`
    :func:`doy`                                        :func:`noisy_empty_approx_set_sfm`                 :func:`transform`
    :func:`e`                                          :func:`none_match`                                 :func:`transform_keys`
    :func:`element_at`                                 :func:`normal_cdf`                                 :func:`transform_values`
    :func:`empty_approx_set`                           :func:`normalize`                                  :func:`trim`
    :func:`ends_with`                                  not                                                :func:`trim_array`
    :func:`enum_key`                                   :func:`parse_datetime`                             :func:`trimmed_mean`
    :func:`eq`                                         :func:`parse_duration`                             :func:`truncate`
    :func:`exp`                                        :func:`parse_presto_data_size`                     :func:`typeof`
    :func:`expand_envelope`                            :func:`pi`                                         :func:`upper`
    :func:`f_cdf`                                      :func:`plus`                                       :func:`url_decode`
    :func:`fail`                                       :func:`poisson_cdf`                                :func:`url_encode`
    :func:`filter`                                     :func:`pow`                                        :func:`url_extract_fragment`
    :func:`find_first`                                 :func:`power`                                      :func:`url_extract_host`
    :func:`find_first_index`                           :func:`quantile_at_value`                          :func:`url_extract_parameter`
    :func:`flatten`                                    :func:`quantiles_at_values`                        :func:`url_extract_path`
    :func:`flatten_geometry_collections`               :func:`quarter`                                    :func:`url_extract_port`
    :func:`floor`                                      :func:`radians`                                    :func:`url_extract_protocol`
    :func:`format_datetime`                            :func:`rand`                                       :func:`url_extract_query`
    :func:`from_base`                                  :func:`random`                                     :func:`uuid`
    :func:`from_base32`                                :func:`reduce`                                     :func:`value_at_quantile`
    :func:`from_base64`                                :func:`regexp_extract`                             :func:`values_at_quantiles`
    :func:`from_base64url`                             :func:`regexp_extract_all`                         :func:`week`
    :func:`from_big_endian_32`                         :func:`regexp_like`                                :func:`week_of_year`
    :func:`from_big_endian_64`                         :func:`regexp_replace`                             :func:`weibull_cdf`
    :func:`from_hex`                                   :func:`regexp_split`                               :func:`width_bucket`
    :func:`from_ieee754_32`                            :func:`remap_keys`                                 :func:`wilson_interval_lower`
    :func:`from_ieee754_64`                            :func:`remove_nulls`                               :func:`wilson_interval_upper`
    :func:`from_iso8601_date`                          :func:`repeat`                                     :func:`word_stem`
    :func:`from_iso8601_timestamp`                     :func:`replace`                                    :func:`xxhash64`
    :func:`from_unixtime`                              :func:`replace_first`                              :func:`xxhash64_internal`
    :func:`from_utf8`                                  :func:`reverse`                                    :func:`year`
    :func:`gamma_cdf`                                  :func:`round`                                      :func:`year_of_week`
    :func:`geometry_as_geojson`                        :func:`rpad`                                       :func:`yow`
    :func:`geometry_from_geojson`                      :func:`rtrim`                                      :func:`zip`
    :func:`geometry_invalid_reason`                    :func:`scale_qdigest`                              :func:`zip_with`
    :func:`geometry_nearest_points`                    :func:`scale_tdigest`
    :func:`geometry_to_bing_tiles`                     :func:`second`
    =================================================  =================================================  =================================================  ==  =================================================  ==  =================================================
