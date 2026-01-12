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
    :func:`abs`                                        :func:`gt`                                         :func:`scale_qdigest`                                  :func:`any_value`                                      :func:`cume_dist`
    :func:`acos`                                       :func:`gte`                                        :func:`scale_tdigest`                                  :func:`approx_distinct`                                :func:`dense_rank`
    :func:`all_keys_match`                             :func:`hamming_distance`                           :func:`second`                                         :func:`approx_most_frequent`                           :func:`first_value`
    :func:`all_match`                                  :func:`hash_counts`                                :func:`secure_rand`                                    :func:`approx_percentile`                              :func:`lag`
    :func:`any_keys_match`                             :func:`hmac_md5`                                   :func:`secure_random`                                  :func:`approx_set`                                     :func:`last_value`
    :func:`any_match`                                  :func:`hmac_sha1`                                  :func:`sequence`                                       :func:`arbitrary`                                      :func:`lead`
    :func:`any_values_match`                           :func:`hmac_sha256`                                :func:`sha1`                                           :func:`array_agg`                                      :func:`nth_value`
    :func:`array_average`                              :func:`hmac_sha512`                                :func:`sha256`                                         :func:`avg`                                            :func:`ntile`
    :func:`array_constructor`                          :func:`hour`                                       :func:`sha512`                                         :func:`bitwise_and_agg`                                :func:`percent_rank`
    :func:`array_cum_sum`                              :func:`infinity`                                   :func:`shuffle`                                        :func:`bitwise_or_agg`                                 :func:`rank`
    :func:`array_distinct`                             :func:`intersection_cardinality`                   :func:`sign`                                           :func:`bitwise_xor_agg`                                :func:`row_number`
    :func:`array_duplicates`                           :func:`inverse_beta_cdf`                           :func:`simplify_geometry`                              :func:`bool_and`
    :func:`array_except`                               :func:`inverse_binomial_cdf`                       :func:`sin`                                            :func:`bool_or`
    :func:`array_frequency`                            :func:`inverse_cauchy_cdf`                         :func:`slice`                                          :func:`checksum`
    :func:`array_has_duplicates`                       :func:`inverse_chi_squared_cdf`                    :func:`split`                                          :func:`classification_fall_out`
    :func:`array_intersect`                            :func:`inverse_f_cdf`                              :func:`split_part`                                     :func:`classification_miss_rate`
    :func:`array_join`                                 :func:`inverse_gamma_cdf`                          :func:`split_to_map`                                   :func:`classification_precision`
    :func:`array_max`                                  :func:`inverse_laplace_cdf`                        :func:`split_to_multimap`                              :func:`classification_recall`
    :func:`array_max_by`                               :func:`inverse_normal_cdf`                         :func:`spooky_hash_v2_32`                              :func:`classification_thresholds`
    :func:`array_min`                                  :func:`inverse_poisson_cdf`                        :func:`spooky_hash_v2_64`                              :func:`convex_hull_agg`
    :func:`array_min_by`                               :func:`inverse_t_cdf`                              :func:`sqrt`                                           :func:`corr`
    :func:`array_normalize`                            :func:`inverse_weibull_cdf`                        :func:`st_area`                                        :func:`count`
    :func:`array_position`                             :func:`ip_prefix`                                  :func:`st_asbinary`                                    :func:`count_if`
    :func:`array_remove`                               :func:`ip_prefix_collapse`                         :func:`st_astext`                                      :func:`covar_pop`
    :func:`array_sort`                                 :func:`ip_prefix_subnets`                          :func:`st_boundary`                                    :func:`covar_samp`
    :func:`array_sort_desc`                            :func:`ip_subnet_max`                              :func:`st_buffer`                                      :func:`entropy`
    :func:`array_subset`                               :func:`ip_subnet_min`                              :func:`st_centroid`                                    :func:`every`
    :func:`array_sum`                                  :func:`ip_subnet_range`                            :func:`st_contains`                                    :func:`geometric_mean`
    :func:`array_sum_propagate_element_null`           :func:`is_finite`                                  :func:`st_convexhull`                                  :func:`geometry_union_agg`
    :func:`array_top_n`                                :func:`is_infinite`                                :func:`st_coorddim`                                    :func:`histogram`
    :func:`array_union`                                :func:`is_json_scalar`                             :func:`st_crosses`                                     :func:`khyperloglog_agg`
    :func:`arrays_overlap`                             :func:`is_nan`                                     :func:`st_difference`                                  :func:`kurtosis`
    :func:`asin`                                       :func:`is_null`                                    :func:`st_dimension`                                   :func:`make_set_digest`
    :func:`at_timezone`                                :func:`is_private_ip`                              :func:`st_disjoint`                                    :func:`map_agg`
    :func:`atan`                                       :func:`is_subnet_of`                               :func:`st_distance`                                    :func:`map_union`
    :func:`atan2`                                      :func:`jaccard_index`                              :func:`st_endpoint`                                    :func:`map_union_sum`
    :func:`beta_cdf`                                   :func:`jarowinkler_similarity`                     :func:`st_envelope`                                    :func:`max`
    :func:`between`                                    :func:`json_array_contains`                        :func:`st_envelopeaspts`                               :func:`max_by`
    :func:`bing_tile`                                  :func:`json_array_get`                             :func:`st_equals`                                      :func:`max_data_size_for_stats`
    :func:`bing_tile_at`                               :func:`json_array_length`                          :func:`st_exteriorring`                                :func:`merge`
    :func:`bing_tile_children`                         :func:`json_extract`                               :func:`st_geometries`                                  :func:`merge_set_digest`
    :func:`bing_tile_coordinates`                      :func:`json_extract_scalar`                        :func:`st_geometryfromtext`                            :func:`min`
    :func:`bing_tile_parent`                           :func:`json_format`                                :func:`st_geometryn`                                   :func:`min_by`
    :func:`bing_tile_polygon`                          :func:`json_parse`                                 :func:`st_geometrytype`                                :func:`multimap_agg`
    :func:`bing_tile_quadkey`                          :func:`json_size`                                  :func:`st_geomfrombinary`                              :func:`noisy_approx_distinct_sfm`
    :func:`bing_tile_zoom_level`                       :func:`key_sampling_percent`                       :func:`st_interiorringn`                               :func:`noisy_approx_set_sfm`
    :func:`bing_tiles_around`                          :func:`l2_norm`                                    :func:`st_interiorrings`                               :func:`noisy_approx_set_sfm_from_index_and_zeros`
    :func:`binomial_cdf`                               :func:`laplace_cdf`                                :func:`st_intersection`                                :func:`noisy_avg_gaussian`
    :func:`bit_count`                                  :func:`last_day_of_month`                          :func:`st_intersects`                                  :func:`noisy_count_gaussian`
    :func:`bit_length`                                 :func:`least`                                      :func:`st_isclosed`                                    :func:`noisy_count_if_gaussian`
    :func:`bitwise_and`                                :func:`length`                                     :func:`st_isempty`                                     :func:`noisy_sum_gaussian`
    :func:`bitwise_arithmetic_shift_right`             :func:`levenshtein_distance`                       :func:`st_isring`                                      :func:`numeric_histogram`
    :func:`bitwise_left_shift`                         :func:`like`                                       :func:`st_issimple`                                    :func:`qdigest_agg`
    :func:`bitwise_logical_shift_right`                :func:`line_interpolate_point`                     :func:`st_isvalid`                                     :func:`reduce_agg`
    :func:`bitwise_not`                                :func:`line_locate_point`                          :func:`st_length`                                      :func:`regr_avgx`
    :func:`bitwise_or`                                 :func:`ln`                                         :func:`st_linefromtext`                                :func:`regr_avgy`
    :func:`bitwise_right_shift`                        :func:`localtime`                                  :func:`st_linestring`                                  :func:`regr_count`
    :func:`bitwise_right_shift_arithmetic`             :func:`log10`                                      :func:`st_multipoint`                                  :func:`regr_intercept`
    :func:`bitwise_shift_left`                         :func:`log2`                                       :func:`st_numgeometries`                               :func:`regr_r2`
    :func:`bitwise_xor`                                :func:`longest_common_prefix`                      :func:`st_numinteriorring`                             :func:`regr_slope`
    :func:`cardinality`                                :func:`lower`                                      :func:`st_numpoints`                                   :func:`regr_sxx`
    :func:`cauchy_cdf`                                 :func:`lpad`                                       :func:`st_overlaps`                                    :func:`regr_sxy`
    :func:`cbrt`                                       :func:`lt`                                         :func:`st_point`                                       :func:`regr_syy`
    :func:`ceil`                                       :func:`lte`                                        :func:`st_pointn`                                      :func:`reservoir_sample`
    :func:`ceiling`                                    :func:`ltrim`                                      :func:`st_points`                                      :func:`set_agg`
    :func:`chi_squared_cdf`                            :func:`map`                                        :func:`st_polygon`                                     :func:`set_union`
    :func:`chr`                                        :func:`map_append`                                 :func:`st_relate`                                      :func:`skewness`
    :func:`clamp`                                      :func:`map_concat`                                 :func:`st_startpoint`                                  :func:`stddev`
    :func:`codepoint`                                  :func:`map_entries`                                :func:`st_symdifference`                               :func:`stddev_pop`
    :func:`combinations`                               :func:`map_except`                                 :func:`st_touches`                                     :func:`stddev_samp`
    :func:`combine_hash_internal`                      :func:`map_filter`                                 :func:`st_union`                                       :func:`sum`
    :func:`concat`                                     :func:`map_from_entries`                           :func:`st_within`                                      :func:`sum_data_size_for_stats`
    :func:`construct_tdigest`                          :func:`map_intersect`                              :func:`st_x`                                           :func:`tdigest_agg`
    :func:`contains`                                   :func:`map_key_exists`                             :func:`st_xmax`                                        :func:`var_pop`
    :func:`cos`                                        :func:`map_keys`                                   :func:`st_xmin`                                        :func:`var_samp`
    :func:`cosh`                                       :func:`map_keys_by_top_n_values`                   :func:`st_y`                                           :func:`variance`
    :func:`cosine_similarity`                          :func:`map_keys_overlap`                           :func:`st_ymax`
    :func:`crc32`                                      :func:`map_normalize`                              :func:`st_ymin`
    :func:`current_date`                               :func:`map_remove_null_values`                     :func:`starts_with`
    :func:`current_timestamp`                          :func:`map_subset`                                 :func:`strpos`
    :func:`current_timezone`                           :func:`map_top_n`                                  :func:`strrpos`
    :func:`date`                                       :func:`map_top_n_keys`                             :func:`subscript`
    :func:`date_add`                                   :func:`map_top_n_values`                           :func:`substr`
    :func:`date_diff`                                  :func:`map_trim_values`                            :func:`substring`
    :func:`date_format`                                :func:`map_update`                                 :func:`t_cdf`
    :func:`date_parse`                                 :func:`map_values`                                 :func:`tan`
    :func:`date_trunc`                                 :func:`map_values_in_range`                        :func:`tanh`
    :func:`day`                                        :func:`map_zip_with`                               :func:`timezone_hour`
    :func:`day_of_month`                               :func:`md5`                                        :func:`timezone_minute`
    :func:`day_of_week`                                :func:`merge_hll`                                  :func:`to_base`
    :func:`day_of_year`                                :func:`merge_khll`                                 :func:`to_base64`
    :func:`degrees`                                    :func:`merge_sfm`                                  :func:`to_base64url`
    :func:`destructure_tdigest`                        :func:`merge_tdigest`                              :func:`to_big_endian_32`
    :func:`distinct_from`                              :func:`millisecond`                                :func:`to_big_endian_64`
    :func:`divide`                                     :func:`minus`                                      :func:`to_geometry`
    :func:`dot_product`                                :func:`minute`                                     :func:`to_hex`
    :func:`dow`                                        :func:`mod`                                        :func:`to_ieee754_32`
    :func:`doy`                                        :func:`month`                                      :func:`to_ieee754_64`
    :func:`e`                                          :func:`multimap_from_entries`                      :func:`to_iso8601`
    :func:`element_at`                                 :func:`multiply`                                   :func:`to_milliseconds`
    :func:`empty_approx_set`                           :func:`murmur3_x64_128`                            :func:`to_spherical_geography`
    :func:`ends_with`                                  :func:`nan`                                        :func:`to_unixtime`
    :func:`enum_key`                                   :func:`negate`                                     :func:`to_utf8`
    :func:`eq`                                         :func:`neq`                                        :func:`trail`
    :func:`exp`                                        :func:`ngrams`                                     :func:`transform`
    :func:`expand_envelope`                            :func:`no_keys_match`                              :func:`transform_keys`
    :func:`f_cdf`                                      :func:`no_values_match`                            :func:`transform_values`
    :func:`fail`                                       :func:`noisy_empty_approx_set_sfm`                 :func:`transform_with_index`
    :func:`filter`                                     :func:`none_match`                                 :func:`trim`
    :func:`find_first`                                 :func:`normal_cdf`                                 :func:`trim_array`
    :func:`find_first_index`                           :func:`normalize`                                  :func:`trimmed_mean`
    :func:`flatten`                                    :func:`now`                                        :func:`truncate`
    :func:`flatten_geometry_collections`               :func:`parse_datetime`                             :func:`typeof`
    :func:`floor`                                      :func:`parse_duration`                             :func:`uniqueness_distribution`
    :func:`fnv1_32`                                    :func:`parse_presto_data_size`                     :func:`upper`
    :func:`fnv1_64`                                    :func:`pi`                                         :func:`url_decode`
    :func:`fnv1a_32`                                   :func:`plus`                                       :func:`url_encode`
    :func:`fnv1a_64`                                   :func:`poisson_cdf`                                :func:`url_extract_fragment`
    :func:`format_datetime`                            :func:`pow`                                        :func:`url_extract_host`
    :func:`from_base`                                  :func:`power`                                      :func:`url_extract_parameter`
    :func:`from_base32`                                :func:`quantile_at_value`                          :func:`url_extract_path`
    :func:`from_base64`                                :func:`quantiles_at_values`                        :func:`url_extract_port`
    :func:`from_base64url`                             :func:`quarter`                                    :func:`url_extract_protocol`
    :func:`from_big_endian_32`                         :func:`radians`                                    :func:`url_extract_query`
    :func:`from_big_endian_64`                         :func:`rand`                                       :func:`uuid`
    :func:`from_hex`                                   :func:`random`                                     :func:`value_at_quantile`
    :func:`from_ieee754_32`                            :func:`reduce`                                     :func:`values_at_quantiles`
    :func:`from_ieee754_64`                            :func:`regexp_extract`                             :func:`week`
    :func:`from_iso8601_date`                          :func:`regexp_extract_all`                         :func:`week_of_year`
    :func:`from_iso8601_timestamp`                     :func:`regexp_like`                                :func:`weibull_cdf`
    :func:`from_unixtime`                              :func:`regexp_replace`                             :func:`width_bucket`
    :func:`from_utf8`                                  :func:`regexp_split`                               :func:`wilson_interval_lower`
    :func:`gamma_cdf`                                  :func:`reidentification_potential`                 :func:`wilson_interval_upper`
    :func:`geometry_as_geojson`                        :func:`remap_keys`                                 :func:`word_stem`
    :func:`geometry_from_geojson`                      :func:`remove_nulls`                               :func:`xxhash64`
    :func:`geometry_invalid_reason`                    :func:`repeat`                                     :func:`xxhash64_internal`
    :func:`geometry_nearest_points`                    :func:`replace`                                    :func:`year`
    :func:`geometry_to_bing_tiles`                     :func:`replace_first`                              :func:`year_of_week`
    :func:`geometry_to_dissolved_bing_tiles`           :func:`reverse`                                    :func:`yow`
    :func:`geometry_union`                             :func:`round`                                      :func:`zip`
    :func:`great_circle_distance`                      :func:`rpad`                                       :func:`zip_with`
    :func:`greatest`                                   :func:`rtrim`
    =================================================  =================================================  =================================================  ==  =================================================  ==  =================================================
