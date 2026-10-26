*******************
October 2026 Update
*******************

Hive Connector
==============

* **Breaking change:** Read non-exact NaN spellings such as ``nan`` and ``NAN``
  as NULL in REAL and DOUBLE columns of delimited text tables. Continue to accept
  ``NaN``, ``+NaN`` and ``-NaN``. Read digitless leading-dot fields such as ``.``
  and ``.e2`` as NULL. :pr:`19589`
