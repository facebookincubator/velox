***********************
Spark Functions
***********************

.. toctree::
    :maxdepth: 2

    functions/spark/math
    functions/spark/bitwise
    functions/spark/comparison
    functions/spark/string
    functions/spark/datetime
    functions/spark/array
    functions/spark/map
    functions/spark/regexp
    functions/spark/binary
    functions/spark/json
    functions/spark/aggregate

Here is a list of all scalar and aggregate Spark functions available in Velox.
Function names link to function descriptions. Check out coverage maps
for :doc:`all <functions/spark/coverage>` functions for broader context.

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

    ==========================  ==========================  ==========================  ==  ==========================
    Scalar Functions                                                                        Aggregate Functions
    ==================================================================================  ==  ==========================
    :func:`abs`                 :func:`get_json_object`     :func:`not`                     :func:`last`
    :func:`add`                 :func:`getarrayitem`        :func:`notequalto`
    :func:`aggregate`           :func:`getmapvalue`         :func:`pmod`
    :func:`array`               :func:`greaterthan`         :func:`power`
    :func:`array_contains`      :func:`greaterthanorequal`  :func:`rand`
    :func:`array_intersect`     :func:`greatest`            :func:`regexp_extract`
    :func:`array_sort`          :func:`hash`                :func:`remainder`
    :func:`ascii`               in                          :func:`replace`
    :func:`between`             :func:`instr`               :func:`rlike`
    :func:`bitwise_and`         :func:`isnotnull`           :func:`round`
    :func:`bitwise_or`          :func:`isnull`              :func:`shiftleft`
    :func:`ceil`                :func:`least`               :func:`shiftright`
    :func:`chr`                 :func:`length`              :func:`size`
    :func:`concat`              :func:`lessthan`            :func:`sort_array`
    :func:`contains`            :func:`lessthanorequal`     :func:`split`
    :func:`divide`              :func:`lower`               :func:`startswith`
    :func:`element_at`          :func:`map`                 :func:`substring`
    :func:`endswith`            :func:`map_filter`          :func:`subtract`
    :func:`equalnullsafe`       :func:`map_from_arrays`     :func:`transform`
    :func:`equalto`             :func:`md5`                 :func:`unaryminus`
    :func:`exp`                 :func:`multiply`            :func:`upper`
    :func:`filter`              :func:`murmur3hash`         :func:`xxhash64`
    :func:`floor`               :func:`named_struct`        :func:`year`
    ==========================  ==========================  ==========================  ==  ==========================

