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
    :func:`$internal$json_string_to_array_cast`        :func:`greatest`                                   :func:`scale_tdigest`                                  :func:`any_value`                                      :func:`cume_dist`
    :func:`$internal$json_string_to_map_cast`          :func:`gt`                                         :func:`second`                                         :func:`approx_distinct`                                :func:`dense_rank`
    :func:`$internal$json_string_to_row_cast`          :func:`gte`                                        :func:`secure_rand`                                    :func:`approx_most_frequent`                           :func:`first_value`
    :func:`$internal$split_to_map`                     :func:`hamming_distance`                           :func:`secure_random`                                  :func:`approx_percentile`                              :func:`lag`
    :func:`abs`                                        :func:`hash_counts`                                :func:`sequence`                                       :func:`approx_set`                                     :func:`last_value`
    :func:`acos`                                       :func:`hmac_md5`                                   :func:`sha1`                                           :func:`arbitrary`                                      :func:`lead`
    :func:`all_keys_match`                             :func:`hmac_sha1`                                  :func:`sha256`                                         :func:`array_agg`                                      :func:`nth_value`
    :func:`all_match`                                  :func:`hmac_sha256`                                :func:`sha512`                                         :func:`avg`                                            :func:`ntile`
    :func:`any_keys_match`                             :func:`hmac_sha512`                                :func:`shuffle`                                        :func:`bitwise_and_agg`                                :func:`percent_rank`
    :func:`any_match`                                  :func:`hour`                                       :func:`sign`                                           :func:`bitwise_or_agg`                                 :func:`rank`
    :func:`any_values_match`                           in                                                 :func:`simplify_geometry`                              :func:`bitwise_xor_agg`                                :func:`row_number`
    :func:`array_average`                              :func:`infinity`                                   :func:`sin`                                            :func:`bool_and`
    :func:`array_constructor`                          :func:`intersection_cardinality`                   :func:`slice`                                          :func:`bool_or`
    :func:`array_cum_sum`                              :func:`inverse_beta_cdf`                           :func:`split`                                          :func:`checksum`
    :func:`array_distinct`                             :func:`inverse_binomial_cdf`                       :func:`split_part`                                     :func:`classification_fall_out`
    :func:`array_duplicates`                           :func:`inverse_cauchy_cdf`                         :func:`split_to_map`                                   :func:`classification_miss_rate`
    :func:`array_except`                               :func:`inverse_chi_squared_cdf`                    :func:`split_to_multimap`                              :func:`classification_precision`
    :func:`array_frequency`                            :func:`inverse_f_cdf`                              :func:`spooky_hash_v2_32`                              :func:`classification_recall`
    :func:`array_has_duplicates`                       :func:`inverse_gamma_cdf`                          :func:`spooky_hash_v2_64`                              :func:`classification_thresholds`
    :func:`array_intersect`                            :func:`inverse_laplace_cdf`                        :func:`sqrt`                                           :func:`convex_hull_agg`
    :func:`array_join`                                 :func:`inverse_normal_cdf`                         :func:`st_area`                                        :func:`corr`
    :func:`array_max`                                  :func:`inverse_poisson_cdf`                        :func:`st_asbinary`                                    :func:`count`
    :func:`array_max_by`                               :func:`inverse_t_cdf`                              :func:`st_astext`                                      :func:`count_if`
    :func:`array_min`                                  :func:`inverse_weibull_cdf`                        :func:`st_boundary`                                    :func:`covar_pop`
    :func:`array_min_by`                               :func:`ip_prefix`                                  :func:`st_buffer`                                      :func:`covar_samp`
    :func:`array_normalize`                            :func:`ip_prefix_collapse`                         :func:`st_centroid`                                    :func:`entropy`
    :func:`array_position`                             :func:`ip_prefix_subnets`                          :func:`st_contains`                                    :func:`every`
    :func:`array_remove`                               :func:`ip_subnet_max`                              :func:`st_convexhull`                                  :func:`geometric_mean`
    :func:`array_sort`                                 :func:`ip_subnet_min`                              :func:`st_coorddim`                                    :func:`geometry_union_agg`
    :func:`array_sort_desc`                            :func:`ip_subnet_range`                            :func:`st_crosses`                                     :func:`histogram`
    :func:`array_subset`                               :func:`is_finite`                                  :func:`st_difference`                                  :func:`khyperloglog_agg`
    :func:`array_sum`                                  :func:`is_infinite`                                :func:`st_dimension`                                   :func:`kurtosis`
    :func:`array_sum_propagate_element_null`           :func:`is_json_scalar`                             :func:`st_disjoint`                                    :func:`make_set_digest`
    :func:`array_top_n`                                :func:`is_nan`                                     :func:`st_distance`                                    :func:`map_agg`
    :func:`array_union`                                :func:`is_null`                                    :func:`st_endpoint`                                    :func:`map_union`
    :func:`arrays_overlap`                             :func:`is_private_ip`                              :func:`st_envelope`                                    :func:`map_union_sum`
    :func:`asin`                                       :func:`is_subnet_of`                               :func:`st_envelopeaspts`                               :func:`max`
    :func:`at_timezone`                                :func:`jaccard_index`                              :func:`st_equals`                                      :func:`max_by`
    :func:`atan`                                       :func:`jarowinkler_similarity`                     :func:`st_exteriorring`                                :func:`max_data_size_for_stats`
    :func:`atan2`                                      :func:`json_array_contains`                        :func:`st_geometries`                                  :func:`merge`
    :func:`beta_cdf`                                   :func:`json_array_get`                             :func:`st_geometryfromtext`                            :func:`merge_set_digest`
    :func:`between`                                    :func:`json_array_length`                          :func:`st_geometryn`                                   :func:`min`
    :func:`bing_tile`                                  :func:`json_extract`                               :func:`st_geometrytype`                                :func:`min_by`
    :func:`bing_tile_at`                               :func:`json_extract_scalar`                        :func:`st_geomfrombinary`                              :func:`multimap_agg`
    :func:`bing_tile_children`                         :func:`json_format`                                :func:`st_interiorringn`                               :func:`noisy_approx_distinct_sfm`
    :func:`bing_tile_coordinates`                      :func:`json_parse`                                 :func:`st_interiorrings`                               :func:`noisy_approx_set_sfm`
    :func:`bing_tile_parent`                           :func:`json_size`                                  :func:`st_intersection`                                :func:`noisy_approx_set_sfm_from_index_and_zeros`
    :func:`bing_tile_polygon`                          :func:`laplace_cdf`                                :func:`st_intersects`                                  :func:`noisy_avg_gaussian`
    :func:`bing_tile_quadkey`                          :func:`last_day_of_month`                          :func:`st_isclosed`                                    :func:`noisy_count_gaussian`
    :func:`bing_tile_zoom_level`                       :func:`least`                                      :func:`st_isempty`                                     :func:`noisy_count_if_gaussian`
    :func:`bing_tiles_around`                          :func:`length`                                     :func:`st_isring`                                      :func:`noisy_sum_gaussian`
    :func:`binomial_cdf`                               :func:`levenshtein_distance`                       :func:`st_issimple`                                    :func:`numeric_histogram`
    :func:`bit_count`                                  :func:`like`                                       :func:`st_isvalid`                                     :func:`qdigest_agg`
    :func:`bit_length`                                 :func:`line_interpolate_point`                     :func:`st_length`                                      :func:`reduce_agg`
    :func:`bitwise_and`                                :func:`line_locate_point`                          :func:`st_linefromtext`                                :func:`regr_avgx`
    :func:`bitwise_arithmetic_shift_right`             :func:`ln`                                         :func:`st_linestring`                                  :func:`regr_avgy`
    :func:`bitwise_left_shift`                         :func:`localtime`                                  :func:`st_multipoint`                                  :func:`regr_count`
    :func:`bitwise_logical_shift_right`                :func:`log10`                                      :func:`st_numgeometries`                               :func:`regr_intercept`
    :func:`bitwise_not`                                :func:`log2`                                       :func:`st_numinteriorring`                             :func:`regr_r2`
    :func:`bitwise_or`                                 :func:`longest_common_prefix`                      :func:`st_numpoints`                                   :func:`regr_slope`
    :func:`bitwise_right_shift`                        :func:`lower`                                      :func:`st_overlaps`                                    :func:`regr_sxx`
    :func:`bitwise_right_shift_arithmetic`             :func:`lpad`                                       :func:`st_point`                                       :func:`regr_sxy`
    :func:`bitwise_shift_left`                         :func:`lt`                                         :func:`st_pointn`                                      :func:`regr_syy`
    :func:`bitwise_xor`                                :func:`lte`                                        :func:`st_points`                                      :func:`reservoir_sample`
    :func:`cardinality`                                :func:`ltrim`                                      :func:`st_polygon`                                     :func:`set_agg`
    :func:`cauchy_cdf`                                 :func:`map`                                        :func:`st_relate`                                      :func:`set_union`
    :func:`cbrt`                                       :func:`map_append`                                 :func:`st_startpoint`                                  :func:`skewness`
    :func:`ceil`                                       :func:`map_concat`                                 :func:`st_symdifference`                               :func:`stddev`
    :func:`ceiling`                                    :func:`map_entries`                                :func:`st_touches`                                     :func:`stddev_pop`
    :func:`chi_squared_cdf`                            :func:`map_except`                                 :func:`st_union`                                       :func:`stddev_samp`
    :func:`chr`                                        :func:`map_filter`                                 :func:`st_within`                                      :func:`sum`
    :func:`clamp`                                      :func:`map_from_entries`                           :func:`st_x`                                           :func:`sum_data_size_for_stats`
    :func:`codepoint`                                  :func:`map_intersect`                              :func:`st_xmax`                                        :func:`tdigest_agg`
    :func:`combinations`                               :func:`map_key_exists`                             :func:`st_xmin`                                        :func:`var_pop`
    :func:`combine_hash_internal`                      :func:`map_keys`                                   :func:`st_y`                                           :func:`var_samp`
    :func:`concat`                                     :func:`map_keys_by_top_n_values`                   :func:`st_ymax`                                        :func:`variance`
    :func:`construct_tdigest`                          :func:`map_keys_overlap`                           :func:`st_ymin`
    :func:`contains`                                   :func:`map_normalize`                              :func:`starts_with`
    :func:`cos`                                        :func:`map_remove_null_values`                     :func:`strpos`
    :func:`cosh`                                       :func:`map_subset`                                 :func:`strrpos`
    :func:`cosine_similarity`                          :func:`map_top_n`                                  :func:`subscript`
    :func:`crc32`                                      :func:`map_top_n_keys`                             :func:`substr`
    :func:`current_date`                               :func:`map_top_n_values`                           :func:`substring`
    :func:`current_timestamp`                          :func:`map_values`                                 :func:`t_cdf`
    :func:`current_timezone`                           :func:`map_zip_with`                               :func:`tan`
    :func:`date`                                       :func:`md5`                                        :func:`tanh`
    :func:`date_add`                                   :func:`merge_hll`                                  :func:`timezone_hour`
    :func:`date_diff`                                  :func:`merge_khll`                                 :func:`timezone_minute`
    :func:`date_format`                                :func:`merge_sfm`                                  :func:`to_base`
    :func:`date_parse`                                 :func:`merge_tdigest`                              :func:`to_base64`
    :func:`date_trunc`                                 :func:`millisecond`                                :func:`to_base64url`
    :func:`day`                                        :func:`minus`                                      :func:`to_big_endian_32`
    :func:`day_of_month`                               :func:`minute`                                     :func:`to_big_endian_64`
    :func:`day_of_week`                                :func:`mod`                                        :func:`to_geometry`
    :func:`day_of_year`                                :func:`month`                                      :func:`to_hex`
    :func:`degrees`                                    :func:`multimap_from_entries`                      :func:`to_ieee754_32`
    :func:`destructure_tdigest`                        :func:`multiply`                                   :func:`to_ieee754_64`
    :func:`distinct_from`                              :func:`murmur3_x64_128`                            :func:`to_iso8601`
    :func:`divide`                                     :func:`nan`                                        :func:`to_milliseconds`
    :func:`dot_product`                                :func:`negate`                                     :func:`to_spherical_geography`
    :func:`dow`                                        :func:`neq`                                        :func:`to_unixtime`
    :func:`doy`                                        :func:`ngrams`                                     :func:`to_utf8`
    :func:`e`                                          :func:`no_keys_match`                              :func:`trail`
    :func:`element_at`                                 :func:`no_values_match`                            :func:`transform`
    :func:`empty_approx_set`                           :func:`noisy_empty_approx_set_sfm`                 :func:`transform_keys`
    :func:`ends_with`                                  :func:`none_match`                                 :func:`transform_values`
    :func:`enum_key`                                   :func:`normal_cdf`                                 :func:`trim`
    :func:`eq`                                         :func:`normalize`                                  :func:`trim_array`
    :func:`exp`                                        not                                                :func:`trimmed_mean`
    :func:`expand_envelope`                            :func:`now`                                        :func:`truncate`
    :func:`f_cdf`                                      :func:`parse_datetime`                             :func:`typeof`
    :func:`fail`                                       :func:`parse_duration`                             :func:`uniqueness_distribution`
    :func:`filter`                                     :func:`parse_presto_data_size`                     :func:`upper`
    :func:`find_first`                                 :func:`pi`                                         :func:`url_decode`
    :func:`find_first_index`                           :func:`plus`                                       :func:`url_encode`
    :func:`flatten`                                    :func:`poisson_cdf`                                :func:`url_extract_fragment`
    :func:`flatten_geometry_collections`               :func:`pow`                                        :func:`url_extract_host`
    :func:`floor`                                      :func:`power`                                      :func:`url_extract_parameter`
    :func:`format_datetime`                            :func:`quantile_at_value`                          :func:`url_extract_path`
    :func:`from_base`                                  :func:`quantiles_at_values`                        :func:`url_extract_port`
    :func:`from_base32`                                :func:`quarter`                                    :func:`url_extract_protocol`
    :func:`from_base64`                                :func:`radians`                                    :func:`url_extract_query`
    :func:`from_base64url`                             :func:`rand`                                       :func:`uuid`
    :func:`from_big_endian_32`                         :func:`random`                                     :func:`value_at_quantile`
    :func:`from_big_endian_64`                         :func:`reduce`                                     :func:`values_at_quantiles`
    :func:`from_hex`                                   :func:`regexp_extract`                             :func:`week`
    :func:`from_ieee754_32`                            :func:`regexp_extract_all`                         :func:`week_of_year`
    :func:`from_ieee754_64`                            :func:`regexp_like`                                :func:`weibull_cdf`
    :func:`from_iso8601_date`                          :func:`regexp_replace`                             :func:`width_bucket`
    :func:`from_iso8601_timestamp`                     :func:`regexp_split`                               :func:`wilson_interval_lower`
    :func:`from_unixtime`                              :func:`reidentification_potential`                 :func:`wilson_interval_upper`
    :func:`from_utf8`                                  :func:`remap_keys`                                 :func:`word_stem`
    :func:`gamma_cdf`                                  :func:`remove_nulls`                               :func:`xxhash64`
    :func:`geometry_as_geojson`                        :func:`repeat`                                     :func:`xxhash64_internal`
    :func:`geometry_from_geojson`                      :func:`replace`                                    :func:`year`
    :func:`geometry_invalid_reason`                    :func:`replace_first`                              :func:`year_of_week`
    :func:`geometry_nearest_points`                    :func:`reverse`                                    :func:`yow`
    :func:`geometry_to_bing_tiles`                     :func:`round`                                      :func:`zip`
    :func:`geometry_to_dissolved_bing_tiles`           :func:`rpad`                                       :func:`zip_with`
    :func:`geometry_union`                             :func:`rtrim`
    :func:`great_circle_distance`                      :func:`scale_qdigest`
    =================================================  =================================================  =================================================  ==  =================================================  ==  =================================================
