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
    :func:`abs`                                        :func:`hash_counts`                                :func:`s2_cell_parent`                                 :func:`any_value`                                      :func:`cume_dist`
    :func:`acos`                                       :func:`hmac_md5`                                   :func:`s2_cell_to_token`                               :func:`approx_distinct`                                :func:`dense_rank`
    :func:`all_keys_match`                             :func:`hmac_sha1`                                  :func:`s2_cells`                                       :func:`approx_most_frequent`                           :func:`first_value`
    :func:`all_match`                                  :func:`hmac_sha256`                                :func:`scale_qdigest`                                  :func:`approx_percentile`                              :func:`lag`
    :func:`any_keys_match`                             :func:`hmac_sha512`                                :func:`scale_tdigest`                                  :func:`approx_set`                                     :func:`last_value`
    :func:`any_match`                                  :func:`hour`                                       :func:`second`                                         :func:`approx_winsorized_mean`                         :func:`lead`
    :func:`any_values_match`                           :func:`infinity`                                   :func:`secure_rand`                                    :func:`arbitrary`                                      :func:`nth_value`
    :func:`array_average`                              :func:`intersection_cardinality`                   :func:`secure_random`                                  :func:`array_agg`                                      :func:`ntile`
    :func:`array_constructor`                          :func:`inverse_beta_cdf`                           :func:`sequence`                                       :func:`avg`                                            :func:`percent_rank`
    :func:`array_cum_sum`                              :func:`inverse_binomial_cdf`                       :func:`sha1`                                           :func:`bitwise_and_agg`                                :func:`rank`
    :func:`array_distinct`                             :func:`inverse_cauchy_cdf`                         :func:`sha256`                                         :func:`bitwise_or_agg`                                 :func:`row_number`
    :func:`array_duplicates`                           :func:`inverse_chi_squared_cdf`                    :func:`sha512`                                         :func:`bitwise_xor_agg`
    :func:`array_except`                               :func:`inverse_f_cdf`                              :func:`shuffle`                                        :func:`bool_and`
    :func:`array_frequency`                            :func:`inverse_gamma_cdf`                          :func:`sign`                                           :func:`bool_or`
    :func:`array_has_duplicates`                       :func:`inverse_laplace_cdf`                        :func:`simplify_geometry`                              :func:`checksum`
    :func:`array_intersect`                            :func:`inverse_normal_cdf`                         :func:`sin`                                            :func:`classification_fall_out`
    :func:`array_join`                                 :func:`inverse_poisson_cdf`                        :func:`slice`                                          :func:`classification_miss_rate`
    :func:`array_max`                                  :func:`inverse_t_cdf`                              :func:`split`                                          :func:`classification_precision`
    :func:`array_max_by`                               :func:`inverse_weibull_cdf`                        :func:`split_part`                                     :func:`classification_recall`
    :func:`array_min`                                  :func:`ip_prefix`                                  :func:`split_to_map`                                   :func:`classification_thresholds`
    :func:`array_min_by`                               :func:`ip_prefix_collapse`                         :func:`split_to_multimap`                              :func:`convex_hull_agg`
    :func:`array_normalize`                            :func:`ip_prefix_masklen`                          :func:`spooky_hash_v2_32`                              :func:`corr`
    :func:`array_position`                             :func:`ip_prefix_subnets`                          :func:`spooky_hash_v2_64`                              :func:`count`
    :func:`array_remove`                               :func:`ip_subnet_max`                              :func:`sqrt`                                           :func:`count_if`
    :func:`array_sort`                                 :func:`ip_subnet_min`                              :func:`st_area`                                        :func:`covar_pop`
    :func:`array_sort_desc`                            :func:`ip_subnet_range`                            :func:`st_asbinary`                                    :func:`covar_samp`
    :func:`array_split_into_chunks`                    :func:`ip_version`                                 :func:`st_astext`                                      :func:`entropy`
    :func:`array_subset`                               :func:`is_finite`                                  :func:`st_boundary`                                    :func:`every`
    :func:`array_sum`                                  :func:`is_infinite`                                :func:`st_buffer`                                      :func:`geometric_mean`
    :func:`array_sum_propagate_element_null`           :func:`is_json_scalar`                             :func:`st_centroid`                                    :func:`geometry_union_agg`
    :func:`array_top_n`                                :func:`is_nan`                                     :func:`st_contains`                                    :func:`histogram`
    :func:`array_union`                                :func:`is_null`                                    :func:`st_convexhull`                                  :func:`khyperloglog_agg`
    :func:`arrays_overlap`                             :func:`is_private_ip`                              :func:`st_coorddim`                                    :func:`kurtosis`
    :func:`asin`                                       :func:`is_subnet_of`                               :func:`st_crosses`                                     :func:`make_set_digest`
    :func:`at_timezone`                                :func:`jaccard_index`                              :func:`st_difference`                                  :func:`map_agg`
    :func:`atan`                                       :func:`jarowinkler_similarity`                     :func:`st_dimension`                                   :func:`map_union`
    :func:`atan2`                                      :func:`json_array_contains`                        :func:`st_disjoint`                                    :func:`map_union_sum`
    :func:`beta_cdf`                                   :func:`json_array_get`                             :func:`st_distance`                                    :func:`max`
    :func:`between`                                    :func:`json_array_length`                          :func:`st_endpoint`                                    :func:`max_by`
    :func:`bing_tile`                                  :func:`json_extract`                               :func:`st_envelope`                                    :func:`max_data_size_for_stats`
    :func:`bing_tile_at`                               :func:`json_extract_scalar`                        :func:`st_envelopeaspts`                               :func:`merge`
    :func:`bing_tile_children`                         :func:`json_format`                                :func:`st_equals`                                      :func:`merge_set_digest`
    :func:`bing_tile_coordinates`                      :func:`json_parse`                                 :func:`st_exteriorring`                                :func:`min`
    :func:`bing_tile_parent`                           :func:`json_size`                                  :func:`st_geometries`                                  :func:`min_by`
    :func:`bing_tile_polygon`                          :func:`key_sampling_percent`                       :func:`st_geometryfromtext`                            :func:`multimap_agg`
    :func:`bing_tile_quadkey`                          :func:`l2_norm`                                    :func:`st_geometryn`                                   :func:`noisy_approx_distinct_sfm`
    :func:`bing_tile_zoom_level`                       :func:`laplace_cdf`                                :func:`st_geometrytype`                                :func:`noisy_approx_set_sfm`
    :func:`bing_tiles_around`                          :func:`last_day_of_month`                          :func:`st_geomfrombinary`                              :func:`noisy_approx_set_sfm_from_index_and_zeros`
    :func:`binomial_cdf`                               :func:`least`                                      :func:`st_interiorringn`                               :func:`noisy_avg_gaussian`
    :func:`bit_count`                                  :func:`length`                                     :func:`st_interiorrings`                               :func:`noisy_count_gaussian`
    :func:`bit_length`                                 :func:`levenshtein_distance`                       :func:`st_intersection`                                :func:`noisy_count_if_gaussian`
    :func:`bitwise_and`                                :func:`like`                                       :func:`st_intersects`                                  :func:`noisy_sum_gaussian`
    :func:`bitwise_arithmetic_shift_right`             :func:`line_interpolate_point`                     :func:`st_isclosed`                                    :func:`numeric_histogram`
    :func:`bitwise_left_shift`                         :func:`line_locate_point`                          :func:`st_isempty`                                     :func:`qdigest_agg`
    :func:`bitwise_logical_shift_right`                :func:`ln`                                         :func:`st_isring`                                      :func:`reduce_agg`
    :func:`bitwise_not`                                :func:`localtime`                                  :func:`st_issimple`                                    :func:`regr_avgx`
    :func:`bitwise_or`                                 :func:`localtimestamp`                             :func:`st_isvalid`                                     :func:`regr_avgy`
    :func:`bitwise_right_shift`                        :func:`log10`                                      :func:`st_length`                                      :func:`regr_count`
    :func:`bitwise_right_shift_arithmetic`             :func:`log2`                                       :func:`st_linefromtext`                                :func:`regr_intercept`
    :func:`bitwise_shift_left`                         :func:`longest_common_prefix`                      :func:`st_linestring`                                  :func:`regr_r2`
    :func:`bitwise_xor`                                :func:`lower`                                      :func:`st_multipoint`                                  :func:`regr_slope`
    :func:`cardinality`                                :func:`lpad`                                       :func:`st_numgeometries`                               :func:`regr_sxx`
    :func:`cauchy_cdf`                                 :func:`lt`                                         :func:`st_numinteriorring`                             :func:`regr_sxy`
    :func:`cbrt`                                       :func:`lte`                                        :func:`st_numpoints`                                   :func:`regr_syy`
    :func:`ceil`                                       :func:`ltrim`                                      :func:`st_overlaps`                                    :func:`reservoir_sample`
    :func:`ceiling`                                    :func:`map`                                        :func:`st_point`                                       :func:`set_agg`
    :func:`chi_squared_cdf`                            :func:`map_append`                                 :func:`st_pointn`                                      :func:`set_union`
    :func:`chr`                                        :func:`map_concat`                                 :func:`st_points`                                      :func:`skewness`
    :func:`clamp`                                      :func:`map_entries`                                :func:`st_polygon`                                     :func:`stddev`
    :func:`codepoint`                                  :func:`map_except`                                 :func:`st_relate`                                      :func:`stddev_pop`
    :func:`combinations`                               :func:`map_filter`                                 :func:`st_startpoint`                                  :func:`stddev_samp`
    :func:`combine_hash_internal`                      :func:`map_from_entries`                           :func:`st_symdifference`                               :func:`sum`
    :func:`concat`                                     :func:`map_intersect`                              :func:`st_touches`                                     :func:`sum_data_size_for_stats`
    :func:`construct_tdigest`                          :func:`map_key_exists`                             :func:`st_union`                                       :func:`tdigest_agg`
    :func:`contains`                                   :func:`map_keys`                                   :func:`st_within`                                      :func:`var_pop`
    :func:`cos`                                        :func:`map_keys_by_top_n_values`                   :func:`st_x`                                           :func:`var_samp`
    :func:`cosh`                                       :func:`map_keys_overlap`                           :func:`st_xmax`                                        :func:`variance`
    :func:`cosine_similarity`                          :func:`map_normalize`                              :func:`st_xmin`                                        :func:`vector_sum`
    :func:`crc32`                                      :func:`map_remove_null_values`                     :func:`st_y`
    :func:`current_date`                               :func:`map_subset`                                 :func:`st_ymax`
    :func:`current_time`                               :func:`map_top_n`                                  :func:`st_ymin`
    :func:`current_timestamp`                          :func:`map_top_n_keys`                             :func:`starts_with`
    :func:`current_timezone`                           :func:`map_top_n_values`                           :func:`strpos`
    :func:`date`                                       :func:`map_trim_values`                            :func:`strrpos`
    :func:`date_add`                                   :func:`map_update`                                 :func:`subscript`
    :func:`date_diff`                                  :func:`map_values`                                 :func:`substr`
    :func:`date_format`                                :func:`map_values_in_range`                        :func:`substring`
    :func:`date_parse`                                 :func:`map_zip_with`                               :func:`t_cdf`
    :func:`date_trunc`                                 :func:`md5`                                        :func:`tan`
    :func:`day`                                        :func:`merge_hll`                                  :func:`tanh`
    :func:`day_of_month`                               :func:`merge_khll`                                 :func:`timezone_hour`
    :func:`day_of_week`                                :func:`merge_sfm`                                  :func:`timezone_minute`
    :func:`day_of_year`                                :func:`merge_tdigest`                              :func:`to_base`
    :func:`degrees`                                    :func:`millisecond`                                :func:`to_base64`
    :func:`destructure_tdigest`                        :func:`minus`                                      :func:`to_base64url`
    :func:`distinct_from`                              :func:`minute`                                     :func:`to_big_endian_32`
    :func:`divide`                                     :func:`mod`                                        :func:`to_big_endian_64`
    :func:`dot_product`                                :func:`month`                                      :func:`to_geometry`
    :func:`dow`                                        :func:`multimap_from_entries`                      :func:`to_hex`
    :func:`doy`                                        :func:`multiply`                                   :func:`to_ieee754_32`
    :func:`e`                                          :func:`murmur3_x64_128`                            :func:`to_ieee754_64`
    :func:`element_at`                                 :func:`nan`                                        :func:`to_iso8601`
    :func:`empty_approx_set`                           :func:`negate`                                     :func:`to_milliseconds`
    :func:`ends_with`                                  :func:`neq`                                        :func:`to_spherical_geography`
    :func:`enum_key`                                   :func:`ngrams`                                     :func:`to_unixtime`
    :func:`eq`                                         :func:`no_keys_match`                              :func:`to_utf8`
    :func:`exp`                                        :func:`no_values_match`                            :func:`trail`
    :func:`expand_envelope`                            :func:`noisy_empty_approx_set_sfm`                 :func:`transform`
    :func:`f_cdf`                                      :func:`none_match`                                 :func:`transform_keys`
    :func:`fail`                                       :func:`normal_cdf`                                 :func:`transform_values`
    :func:`filter`                                     :func:`normalize`                                  :func:`transform_with_index`
    :func:`find_first`                                 :func:`now`                                        :func:`trim`
    :func:`find_first_index`                           :func:`parse_datetime`                             :func:`trim_array`
    :func:`flatten`                                    :func:`parse_duration`                             :func:`trimmed_mean`
    :func:`flatten_geometry_collections`               :func:`parse_presto_data_size`                     :func:`truncate`
    :func:`floor`                                      :func:`pi`                                         :func:`typeof`
    :func:`fnv1_32`                                    :func:`plus`                                       :func:`uniqueness_distribution`
    :func:`fnv1_64`                                    :func:`pmod`                                       :func:`upper`
    :func:`fnv1a_32`                                   :func:`poisson_cdf`                                :func:`url_decode`
    :func:`fnv1a_64`                                   :func:`pow`                                        :func:`url_encode`
    :func:`format_datetime`                            :func:`power`                                      :func:`url_extract_fragment`
    :func:`from_base`                                  :func:`quantile_at_value`                          :func:`url_extract_host`
    :func:`from_base32`                                :func:`quantiles_at_values`                        :func:`url_extract_parameter`
    :func:`from_base64`                                :func:`quarter`                                    :func:`url_extract_path`
    :func:`from_base64url`                             :func:`radians`                                    :func:`url_extract_port`
    :func:`from_big_endian_32`                         :func:`rand`                                       :func:`url_extract_protocol`
    :func:`from_big_endian_64`                         :func:`random`                                     :func:`url_extract_query`
    :func:`from_hex`                                   :func:`reduce`                                     :func:`uuid`
    :func:`from_ieee754_32`                            :func:`regexp_extract`                             :func:`value_at_quantile`
    :func:`from_ieee754_64`                            :func:`regexp_extract_all`                         :func:`values_at_quantiles`
    :func:`from_iso8601_date`                          :func:`regexp_like`                                :func:`week`
    :func:`from_iso8601_timestamp`                     :func:`regexp_replace`                             :func:`week_of_year`
    :func:`from_unixtime`                              :func:`regexp_split`                               :func:`weibull_cdf`
    :func:`from_utf8`                                  :func:`reidentification_potential`                 :func:`width_bucket`
    :func:`gamma_cdf`                                  :func:`remap_keys`                                 :func:`wilson_interval_lower`
    :func:`geometry_as_geojson`                        :func:`remove_nulls`                               :func:`wilson_interval_upper`
    :func:`geometry_from_geojson`                      :func:`repeat`                                     :func:`winsorized_mean`
    :func:`geometry_invalid_reason`                    :func:`replace`                                    :func:`word_stem`
    :func:`geometry_nearest_points`                    :func:`replace_first`                              :func:`xxhash64`
    :func:`geometry_to_bing_tiles`                     :func:`reverse`                                    :func:`xxhash64_internal`
    :func:`geometry_to_dissolved_bing_tiles`           :func:`round`                                      :func:`year`
    :func:`geometry_union`                             :func:`rpad`                                       :func:`year_of_week`
    :func:`great_circle_distance`                      :func:`rtrim`                                      :func:`yow`
    :func:`greatest`                                   :func:`s2_cell_area_sq_km`                         :func:`zip`
    :func:`gt`                                         :func:`s2_cell_contains`                           :func:`zip_with`
    :func:`gte`                                        :func:`s2_cell_from_token`
    :func:`hamming_distance`                           :func:`s2_cell_level`
    =================================================  =================================================  =================================================  ==  =================================================  ==  =================================================
