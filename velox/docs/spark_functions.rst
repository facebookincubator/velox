***********************
Spark Functions
***********************

.. toctree::
    :maxdepth: 1

    functions/spark/math
    functions/spark/bitwise
    functions/spark/comparison
    functions/spark/string
    functions/spark/struct
    functions/spark/datetime
    functions/spark/array
    functions/spark/map
    functions/spark/regexp
    functions/spark/binary
    functions/spark/json
    functions/spark/aggregate

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

    ================================  ================================  ================================  ==  ================================
    Scalar Functions                                                                                          Aggregate Functions
    ====================================================================================================  ==  ================================
    :spark:func:`abs`                 :spark:func:`get_json_object`     :spark:func:`power`                   :spark:func:`last`
    :spark:func:`add`                 :spark:func:`greaterthan`         :spark:func:`rand`
    :spark:func:`aggregate`           :spark:func:`greaterthanorequal`  :spark:func:`regexp_extract`
    :spark:func:`array`               :spark:func:`greatest`            :spark:func:`remainder`
    :spark:func:`array_contains`      :spark:func:`hash`                :spark:func:`replace`
    :spark:func:`array_intersect`     :spark:func:`in`                  :spark:func:`rlike`
    :spark:func:`array_sort`          :spark:func:`instr`               :spark:func:`round`
    :spark:func:`ascii`               :spark:func:`isnotnull`           :spark:func:`sha1`
    :spark:func:`between`             :spark:func:`isnull`              :spark:func:`sha2`
    :spark:func:`bitwise_and`         :spark:func:`least`               :spark:func:`shiftleft`
    :spark:func:`bitwise_or`          :spark:func:`length`              :spark:func:`shiftright`
    :spark:func:`ceil`                :spark:func:`lessthan`            :spark:func:`size`
    :spark:func:`chr`                 :spark:func:`lessthanorequal`     :spark:func:`sort_array`
    :spark:func:`concat`              :spark:func:`lower`               :spark:func:`split`
    :spark:func:`contains`            :spark:func:`map`                 :spark:func:`startswith`
    :spark:func:`divide`              :spark:func:`map_filter`          :spark:func:`substring`
    :spark:func:`element_at`          :spark:func:`map_from_arrays`     :spark:func:`subtract`
    :spark:func:`endswith`            :spark:func:`md5`                 :spark:func:`transform`
    :spark:func:`equalnullsafe`       :spark:func:`multiply`            :spark:func:`unaryminus`
    :spark:func:`equalto`             :spark:func:`named_struct`        :spark:func:`upper`
    :spark:func:`exp`                 :spark:func:`not`                 :spark:func:`xxhash64`
    :spark:func:`filter`              :spark:func:`notequalto`          :spark:func:`year`
    :spark:func:`floor`               :spark:func:`pmod`
    ================================  ================================  ================================  ==  ================================


