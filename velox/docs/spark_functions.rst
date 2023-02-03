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
    :spark:func:`abs`                 :spark:func:`floor`               :spark:func:`notequalto`              :spark:func:`last`
    :spark:func:`add`                 :spark:func:`get_json_object`     :spark:func:`pmod`
    :spark:func:`aggregate`           :spark:func:`greaterthan`         :spark:func:`power`
    :spark:func:`array`               :spark:func:`greaterthanorequal`  :spark:func:`rand`
    :spark:func:`array_contains`      :spark:func:`greatest`            :spark:func:`regexp_extract`
    :spark:func:`array_intersect`     :spark:func:`hash`                :spark:func:`remainder`
    :spark:func:`array_sort`          :spark:func:`in`                  :spark:func:`replace`
    :spark:func:`ascii`               :spark:func:`instr`               :spark:func:`rlike`
    :spark:func:`between`             :spark:func:`isnotnull`           :spark:func:`round`
    :spark:func:`bitwise_and`         :spark:func:`isnull`              :spark:func:`shiftleft`
    :spark:func:`bitwise_or`          :spark:func:`least`               :spark:func:`shiftright`
    :spark:func:`ceil`                :spark:func:`length`              :spark:func:`size`
    :spark:func:`chr`                 :spark:func:`lessthan`            :spark:func:`sort_array`
    :spark:func:`concat`              :spark:func:`lessthanorequal`     :spark:func:`split`
    :spark:func:`contains`            :spark:func:`lower`               :spark:func:`startswith`
    :spark:func:`divide`              :spark:func:`map`                 :spark:func:`substring`
    :spark:func:`element_at`          :spark:func:`map_filter`          :spark:func:`subtract`
    :spark:func:`endswith`            :spark:func:`map_from_arrays`     :spark:func:`transform`
    :spark:func:`equalnullsafe`       :spark:func:`md5`                 :spark:func:`unaryminus`
    :spark:func:`equalto`             :spark:func:`multiply`            :spark:func:`upper`
    :spark:func:`exp`                 :spark:func:`named_struct`        :spark:func:`xxhash64`
    :spark:func:`filter`              :spark:func:`not`                 :spark:func:`year`
    ================================  ================================  ================================  ==  ================================

