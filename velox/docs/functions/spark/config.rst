========================
SparkRegistration Configuration properties
========================

Generic Configuration
---------------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - allowPrecisionLoss
     - bool
     - true
     - When true, establishing the result type of an arithmetic operation according to Hive behavior and SQL ANSI 2011 specification, i.e.
       rounding the decimal part of the result if an exact representation is not
       possible. Otherwise, NULL is returned when the actual result cannot be represented with the calculated decimal type. Now we support add,
       subtract, multiply and divide operations.