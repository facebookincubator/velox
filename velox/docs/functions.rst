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
    :func:`abs`                                        :func:`hash_counts`                                :func:`s2_cell_from_token`                             :func:`any_value`                                      :func:`cume_dist`
    :func:`acos`                                       :func:`hmac_md5`                                   :func:`s2_cell_level`                                  :func:`approx_distinct`                                :func:`dense_rank`
    :func:`all_keys_match`                             :func:`hmac_sha1`                                  :func:`s2_cell_parent`                                 :func:`approx_most_frequent`                           :func:`first_value`
    :func:`all_match`                                  :func:`hmac_sha256`                                :func:`s2_cell_to_token`                               :func:`approx_percentile`                              :func:`lag`
    :func:`any_keys_match`                             :func:`hmac_sha512`                                :func:`s2_cells`                                       :func:`approx_set`                                     :func:`last_value`
    :func:`any_match`                                  :func:`hour`                                       :func:`scale_qdigest`                                  :func:`approx_winsorized_mean`                         :func:`lead`
    :func:`any_values_match`                           :func:`infinity`                                   :func:`scale_tdigest`                                  :func:`arbitrary`                                      :func:`nth_value`
    :func:`array_average`                              :func:`intersection_cardinality`                   :func:`second`                                         :func:`array_agg`                                      :func:`ntile`
    :func:`array_constructor`                          :func:`inverse_beta_cdf`                           :func:`secure_rand`                                    :func:`avg`                                            :func:`percent_rank`
    :func:`array_cum_sum`                              :func:`inverse_binomial_cdf`                       :func:`secure_random`                                  :func:`bitwise_and_agg`                                :func:`rank`
    :func:`array_distinct`                             :func:`inverse_cauchy_cdf`                         :func:`sequence`                                       :func:`bitwise_or_agg`                                 :func:`row_number`
    :func:`array_duplicates`                           :func:`inverse_chi_squared_cdf`                    :func:`sha1`                                           :func:`bitwise_xor_agg`
    :func:`array_except`                               :func:`inverse_f_cdf`                              :func:`sha256`                                         :func:`bool_and`
    :func:`array_frequency`                            :func:`inverse_gamma_cdf`                          :func:`sha512`                                         :func:`bool_or`
    :func:`array_has_duplicates`                       :func:`inverse_laplace_cdf`                        :func:`shuffle`                                        :func:`checksum`
    :func:`array_intersect`                            :func:`inverse_normal_cdf`                         :func:`sign`                                           :func:`classification_fall_out`
    :func:`array_join`                                 :func:`inverse_poisson_cdf`                        :func:`simplify_geometry`                              :func:`classification_miss_rate`
    :func:`array_max`                                  :func:`inverse_t_cdf`                              :func:`sin`                                            :func:`classification_precision`
    :func:`array_max_by`                               :func:`inverse_weibull_cdf`                        :func:`slice`                                          :func:`classification_recall`
    :func:`array_min`                                  :func:`ip_prefix`                                  :func:`split`                                          :func:`classification_thresholds`
    :func:`array_min_by`                               :func:`ip_prefix_collapse`                         :func:`split_part`                                     :func:`convex_hull_agg`
    :func:`array_normalize`                            :func:`ip_prefix_masklen`                          :func:`split_to_map`                                   :func:`corr`
    :func:`array_position`                             :func:`ip_prefix_subnets`                          :func:`split_to_multimap`                              :func:`count`
    :func:`array_remove`                               :func:`ip_subnet_max`                              :func:`spooky_hash_v2_32`                              :func:`count_if`
    :func:`array_sort`                                 :func:`ip_subnet_min`                              :func:`spooky_hash_v2_64`                              :func:`covar_pop`
    :func:`array_sort_desc`                            :func:`ip_subnet_range`                            :func:`sqrt`                                           :func:`covar_samp`
    :func:`array_split_into_chunks`                    :func:`ip_version`                                 :func:`st_area`                                        :func:`entropy`
    :func:`array_subset`                               :func:`is_finite`                                  :func:`st_asbinary`                                    :func:`every`
    :func:`array_sum`                                  :func:`is_infinite`                                :func:`st_astext`                                      :func:`geometric_mean`
    :func:`array_sum_propagate_element_null`           :func:`is_json_scalar`                             :func:`st_boundary`                                    :func:`geometry_union_agg`
    :func:`array_top_n`                                :func:`is_nan`                                     :func:`st_buffer`                                      :func:`histogram`
    :func:`array_union`                                :func:`is_null`                                    :func:`st_centroid`                                    :func:`khyperloglog_agg`
    :func:`arrays_overlap`                             :func:`is_private_ip`                              :func:`st_contains`                                    :func:`kurtosis`
    :func:`asin`                                       :func:`is_subnet_of`                               :func:`st_convexhull`                                  :func:`make_set_digest`
    :func:`at_timezone`                                :func:`jaccard_index`                              :func:`st_coorddim`                                    :func:`map_agg`
    :func:`at_timezone_convert`                        :func:`jarowinkler_similarity`                     :func:`st_crosses`                                     :func:`map_union`
    :func:`atan`                                       :func:`json_array_contains`                        :func:`st_difference`                                  :func:`map_union_sum`
    :func:`atan2`                                      :func:`json_array_get`                             :func:`st_dimension`                                   :func:`max`
    :func:`beta_cdf`                                   :func:`json_array_length`                          :func:`st_disjoint`                                    :func:`max_by`
    :func:`between`                                    :func:`json_extract`                               :func:`st_distance`                                    :func:`max_data_size_for_stats`
    :func:`bing_tile`                                  :func:`json_extract_scalar`                        :func:`st_endpoint`                                    :func:`merge`
    :func:`bing_tile_at`                               :func:`json_format`                                :func:`st_envelope`                                    :func:`merge_set_digest`
    :func:`bing_tile_children`                         :func:`json_parse`                                 :func:`st_envelopeaspts`                               :func:`min`
    :func:`bing_tile_coordinates`                      :func:`json_size`                                  :func:`st_equals`                                      :func:`min_by`
    :func:`bing_tile_parent`                           :func:`key_sampling_percent`                       :func:`st_exteriorring`                                :func:`multimap_agg`
    :func:`bing_tile_polygon`                          :func:`l2_norm`                                    :func:`st_geometries`                                  :func:`noisy_approx_distinct_sfm`
    :func:`bing_tile_quadkey`                          :func:`l2_squared`                                 :func:`st_geometryfromtext`                            :func:`noisy_approx_set_sfm`
    :func:`bing_tile_zoom_level`                       :func:`laplace_cdf`                                :func:`st_geometryn`                                   :func:`noisy_approx_set_sfm_from_index_and_zeros`
    :func:`bing_tiles_around`                          :func:`last_day_of_month`                          :func:`st_geometrytype`                                :func:`noisy_avg_gaussian`
    :func:`binomial_cdf`                               :func:`least`                                      :func:`st_geomfrombinary`                              :func:`noisy_count_gaussian`
    :func:`bit_count`                                  :func:`length`                                     :func:`st_interiorringn`                               :func:`noisy_count_if_gaussian`
    :func:`bit_length`                                 :func:`levenshtein_distance`                       :func:`st_interiorrings`                               :func:`noisy_sum_gaussian`
    :func:`bitwise_and`                                :func:`like`                                       :func:`st_intersection`                                :func:`numeric_histogram`
    :func:`bitwise_arithmetic_shift_right`             :func:`line_interpolate_point`                     :func:`st_intersects`                                  :func:`qdigest_agg`
    :func:`bitwise_left_shift`                         :func:`line_locate_point`                          :func:`st_isclosed`                                    :func:`reduce_agg`
    :func:`bitwise_logical_shift_right`                :func:`ln`                                         :func:`st_isempty`                                     :func:`regr_avgx`
    :func:`bitwise_not`                                :func:`localtime`                                  :func:`st_isring`                                      :func:`regr_avgy`
    :func:`bitwise_or`                                 :func:`localtimestamp`                             :func:`st_issimple`                                    :func:`regr_count`
    :func:`bitwise_right_shift`                        :func:`log10`                                      :func:`st_isvalid`                                     :func:`regr_intercept`
    :func:`bitwise_right_shift_arithmetic`             :func:`log2`                                       :func:`st_length`                                      :func:`regr_r2`
    :func:`bitwise_shift_left`                         :func:`longest_common_prefix`                      :func:`st_linefromtext`                                :func:`regr_slope`
    :func:`bitwise_xor`                                :func:`lower`                                      :func:`st_linestring`                                  :func:`regr_sxx`
    :func:`cardinality`                                :func:`lpad`                                       :func:`st_multipoint`                                  :func:`regr_sxy`
    :func:`cauchy_cdf`                                 :func:`lt`                                         :func:`st_numgeometries`                               :func:`regr_syy`
    :func:`cbrt`                                       :func:`lte`                                        :func:`st_numinteriorring`                             :func:`reservoir_sample`
    :func:`ceil`                                       :func:`ltrim`                                      :func:`st_numpoints`                                   :func:`set_agg`
    :func:`ceiling`                                    :func:`map`                                        :func:`st_overlaps`                                    :func:`set_union`
    :func:`chi_squared_cdf`                            :func:`map_append`                                 :func:`st_point`                                       :func:`skewness`
    :func:`chr`                                        :func:`map_concat`                                 :func:`st_pointn`                                      :func:`stddev`
    :func:`clamp`                                      :func:`map_entries`                                :func:`st_points`                                      :func:`stddev_pop`
    :func:`codepoint`                                  :func:`map_except`                                 :func:`st_polygon`                                     :func:`stddev_samp`
    :func:`combinations`                               :func:`map_filter`                                 :func:`st_relate`                                      :func:`sum`
    :func:`combine_hash_internal`                      :func:`map_from_entries`                           :func:`st_startpoint`                                  :func:`sum_data_size_for_stats`
    :func:`concat`                                     :func:`map_intersect`                              :func:`st_symdifference`                               :func:`tdigest_agg`
    :func:`construct_tdigest`                          :func:`map_key_exists`                             :func:`st_touches`                                     :func:`var_pop`
    :func:`contains`                                   :func:`map_keys`                                   :func:`st_union`                                       :func:`var_samp`
    :func:`cos`                                        :func:`map_keys_by_top_n_values`                   :func:`st_within`                                      :func:`variance`
    :func:`cosh`                                       :func:`map_keys_overlap`                           :func:`st_x`                                           :func:`vector_sum`
    :func:`cosine_similarity`                          :func:`map_normalize`                              :func:`st_xmax`
    :func:`crc32`                                      :func:`map_remove_null_values`                     :func:`st_xmin`
    :func:`current_date`                               :func:`map_subset`                                 :func:`st_y`
    :func:`current_time`                               :func:`map_subset_key_in_range`                    :func:`st_ymax`
    :func:`current_timestamp`                          :func:`map_top_n`                                  :func:`st_ymin`
    :func:`current_timezone`                           :func:`map_top_n_keys`                             :func:`starts_with`
    :func:`date`                                       :func:`map_top_n_values`                           :func:`strpos`
    :func:`date_add`                                   :func:`map_trim_values`                            :func:`strrpos`
    :func:`date_diff`                                  :func:`map_update`                                 :func:`subscript`
    :func:`date_format`                                :func:`map_values`                                 :func:`substr`
    :func:`date_parse`                                 :func:`map_values_all_match`                       :func:`substring`
    :func:`date_trunc`                                 :func:`map_values_any_match`                       :func:`t_cdf`
    :func:`day`                                        :func:`map_values_in_range`                        :func:`tan`
    :func:`day_of_month`                               :func:`map_values_none_match`                      :func:`tanh`
    :func:`day_of_week`                                :func:`map_zip_with`                               :func:`timezone_hour`
    :func:`day_of_year`                                :func:`md5`                                        :func:`timezone_minute`
    :func:`degrees`                                    :func:`merge_hll`                                  :func:`to_base`
    :func:`destructure_tdigest`                        :func:`merge_khll`                                 :func:`to_base64`
    :func:`distinct_from`                              :func:`merge_sfm`                                  :func:`to_base64url`
    :func:`divide`                                     :func:`merge_tdigest`                              :func:`to_big_endian_32`
    :func:`dot_product`                                :func:`millisecond`                                :func:`to_big_endian_64`
    :func:`dow`                                        :func:`minus`                                      :func:`to_geometry`
    :func:`doy`                                        :func:`minute`                                     :func:`to_hex`
    :func:`e`                                          :func:`mod`                                        :func:`to_ieee754_32`
    :func:`element_at`                                 :func:`month`                                      :func:`to_ieee754_64`
    :func:`empty_approx_set`                           :func:`multimap_from_entries`                      :func:`to_iso8601`
    :func:`ends_with`                                  :func:`multiply`                                   :func:`to_milliseconds`
    :func:`enum_key`                                   :func:`murmur3_x64_128`                            :func:`to_spherical_geography`
    :func:`eq`                                         :func:`nan`                                        :func:`to_unixtime`
    :func:`exp`                                        :func:`negate`                                     :func:`to_utf8`
    :func:`expand_envelope`                            :func:`neq`                                        :func:`trail`
    :func:`f_cdf`                                      :func:`ngrams`                                     :func:`transform`
    :func:`fail`                                       :func:`no_keys_match`                              :func:`transform_keys`
    :func:`filter`                                     :func:`no_values_match`                            :func:`transform_values`
    :func:`find_first`                                 :func:`noisy_empty_approx_set_sfm`                 :func:`transform_with_index`
    :func:`find_first_index`                           :func:`none_match`                                 :func:`trim`
    :func:`flatten`                                    :func:`normal_cdf`                                 :func:`trim_array`
    :func:`flatten_geometry_collections`               :func:`normalize`                                  :func:`trimmed_mean`
    :func:`floor`                                      :func:`now`                                        :func:`truncate`
    :func:`fnv1_32`                                    :func:`parse_datetime`                             :func:`typeof`
    :func:`fnv1_64`                                    :func:`parse_duration`                             :func:`uniqueness_distribution`
    :func:`fnv1a_32`                                   :func:`parse_presto_data_size`                     :func:`upper`
    :func:`fnv1a_64`                                   :func:`pi`                                         :func:`url_decode`
    :func:`format_datetime`                            :func:`plus`                                       :func:`url_encode`
    :func:`from_base`                                  :func:`pmod`                                       :func:`url_extract_fragment`
    :func:`from_base32`                                :func:`poisson_cdf`                                :func:`url_extract_host`
    :func:`from_base64`                                :func:`pow`                                        :func:`url_extract_parameter`
    :func:`from_base64url`                             :func:`power`                                      :func:`url_extract_path`
    :func:`from_big_endian_32`                         :func:`quantile_at_value`                          :func:`url_extract_port`
    :func:`from_big_endian_64`                         :func:`quantiles_at_values`                        :func:`url_extract_protocol`
    :func:`from_hex`                                   :func:`quarter`                                    :func:`url_extract_query`
    :func:`from_ieee754_32`                            :func:`radians`                                    :func:`uuid`
    :func:`from_ieee754_64`                            :func:`rand`                                       :func:`value_at_quantile`
    :func:`from_iso8601_date`                          :func:`random`                                     :func:`values_at_quantiles`
    :func:`from_iso8601_timestamp`                     :func:`reduce`                                     :func:`week`
    :func:`from_unixtime`                              :func:`regexp_extract`                             :func:`week_of_year`
    :func:`from_utf8`                                  :func:`regexp_extract_all`                         :func:`weibull_cdf`
    :func:`gamma_cdf`                                  :func:`regexp_like`                                :func:`width_bucket`
    :func:`geometry_as_geojson`                        :func:`regexp_replace`                             :func:`wilson_interval_lower`
    :func:`geometry_from_geojson`                      :func:`regexp_split`                               :func:`wilson_interval_upper`
    :func:`geometry_invalid_reason`                    :func:`reidentification_potential`                 :func:`winsorized_mean`
    :func:`geometry_nearest_points`                    :func:`remap_keys`                                 :func:`word_stem`
    :func:`geometry_to_bing_tiles`                     :func:`remove_nulls`                               :func:`xxhash128`
    :func:`geometry_to_dissolved_bing_tiles`           :func:`repeat`                                     :func:`xxhash64`
    :func:`geometry_union`                             :func:`replace`                                    :func:`xxhash64_internal`
    :func:`google_polyline_decode`                     :func:`replace_first`                              :func:`year`
    :func:`google_polyline_encode`                     :func:`reverse`                                    :func:`year_of_week`
    :func:`great_circle_distance`                      :func:`round`                                      :func:`yow`
    :func:`greatest`                                   :func:`rpad`                                       :func:`zip`
    :func:`gt`                                         :func:`rtrim`                                      :func:`zip_with`
    :func:`gte`                                        :func:`s2_cell_area_sq_km`
    :func:`hamming_distance`                           :func:`s2_cell_contains`
    =================================================  =================================================  =================================================  ==  =================================================  ==  =================================================
