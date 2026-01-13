=================
Function Coverage
=================

Here is a list of all scalar, aggregate, and window functions from Presto, with functions that are available in Velox highlighted.

.. raw:: html

    <style>
    div.body {max-width: 1300px;}
    table.coverage th {background-color: lightblue; text-align: center;}
    table.coverage td:nth-child(6) {background-color: lightblue;}
    table.coverage td:nth-child(8) {background-color: lightblue;}
    table.coverage tr:nth-child(1) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(1) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(1) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(1) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(1) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(1) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(1) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(2) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(2) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(2) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(2) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(2) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(2) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(2) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(3) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(3) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(3) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(3) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(3) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(3) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(3) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(4) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(4) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(4) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(4) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(4) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(4) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(5) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(5) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(5) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(5) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(5) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(5) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(5) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(6) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(6) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(6) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(6) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(6) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(6) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(6) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(7) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(7) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(7) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(7) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(7) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(7) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(8) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(8) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(8) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(8) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(8) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(8) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(8) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(9) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(9) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(9) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(9) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(9) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(9) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(9) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(10) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(10) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(10) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(10) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(10) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(10) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(10) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(11) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(11) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(11) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(11) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(11) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(11) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(11) td:nth-child(9) {background-color: #6BA81E;}
    table.coverage tr:nth-child(12) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(12) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(12) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(12) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(12) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(14) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(14) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(14) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(14) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(14) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(14) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(15) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(15) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(15) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(15) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(15) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(16) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(16) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(16) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(16) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(16) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(17) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(17) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(17) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(17) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(17) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(17) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(18) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(18) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(18) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(18) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(18) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(18) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(19) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(19) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(19) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(19) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(19) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(20) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(20) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(20) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(20) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(20) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(20) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(21) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(21) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(21) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(21) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(21) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(21) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(22) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(22) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(22) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(22) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(22) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(22) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(23) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(23) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(23) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(23) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(23) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(23) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(24) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(24) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(24) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(24) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(24) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(25) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(25) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(25) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(25) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(25) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(25) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(26) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(26) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(26) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(26) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(26) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(29) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(29) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(29) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(29) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(29) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(29) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(30) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(30) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(30) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(30) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(30) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(30) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(31) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(31) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(31) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(31) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(31) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(34) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(34) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(34) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(34) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(35) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(35) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(35) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(35) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(39) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(39) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(39) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(39) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(39) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(39) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(40) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(40) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(40) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(40) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(40) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(40) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(41) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(41) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(41) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(41) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(41) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(41) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(42) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(42) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(42) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(42) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(42) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(42) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(43) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(43) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(43) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(43) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(43) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(43) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(44) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(44) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(44) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(44) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(44) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(45) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(45) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(45) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(45) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(45) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(45) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(46) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(46) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(46) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(46) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(46) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(46) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(47) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(47) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(47) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(47) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(47) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(47) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(48) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(48) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(48) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(48) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(48) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(48) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(49) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(49) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(49) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(49) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(49) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(49) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(50) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(50) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(50) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(50) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(50) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(50) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(51) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(51) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(51) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(51) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(51) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(51) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(52) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(52) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(52) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(52) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(52) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(52) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(53) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(53) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(53) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(53) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(53) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(53) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(55) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(55) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(55) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(55) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(55) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(55) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(56) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(56) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(56) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(56) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(56) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(56) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(57) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(57) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(57) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(57) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(57) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(58) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(58) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(58) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(58) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(58) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(59) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(59) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(59) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(59) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(59) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(59) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(60) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(60) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(60) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(60) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(60) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(60) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(61) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(61) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(61) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(61) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(61) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(61) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(62) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(62) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(62) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(62) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(62) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(62) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(63) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(63) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(63) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(63) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(63) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(63) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(64) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(64) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(64) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(64) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(64) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(64) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(65) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(65) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(65) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(65) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(65) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(66) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(66) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(66) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(66) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(66) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(66) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(67) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(67) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(67) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(67) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(68) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(68) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(68) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(68) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(68) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(69) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(69) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(69) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(69) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(69) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(69) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(70) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(70) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(70) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(70) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(70) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(71) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(71) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(71) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(71) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(71) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(72) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(72) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(72) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(72) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(72) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(72) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(73) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(73) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(73) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(73) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(73) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(73) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(74) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(74) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(74) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(74) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(74) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(74) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(75) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(75) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(75) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(75) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(75) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(75) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(76) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(76) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(76) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(76) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(76) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(76) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(77) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(77) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(77) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(77) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(77) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(77) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(78) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(78) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(78) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(78) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(78) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(78) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(79) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(79) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(79) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(79) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(79) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(80) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(80) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(80) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(80) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(81) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(81) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(81) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(81) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(82) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(82) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(82) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(82) td:nth-child(4) {background-color: #6BA81E;}
    </style>

.. table::
    :widths: auto
    :class: coverage

    ========================================  ========================================  ========================================  ========================================  ========================================  ==  ========================================  ==  ========================================
    Scalar Functions                                                                                                                                                                                                      Aggregate Functions                           Window Functions
    ================================================================================================================================================================================================================  ==  ========================================  ==  ========================================
    :func:`abs`                               :func:`date_parse`                        :func:`is_json_scalar`                    :func:`regexp_like`                       :func:`st_numgeometries`                      :func:`approx_distinct`                       :func:`cume_dist`
    :func:`acos`                              :func:`date_trunc`                        :func:`is_nan`                            :func:`regexp_replace`                    :func:`st_numinteriorring`                    :func:`approx_most_frequent`                  :func:`dense_rank`
    :func:`all_match`                         :func:`day`                               :func:`is_private_ip`                     :func:`regexp_split`                      :func:`st_numpoints`                          :func:`approx_percentile`                     :func:`first_value`
    :func:`any_keys_match`                    :func:`day_of_month`                      :func:`is_subnet_of`                      regress                                   :func:`st_overlaps`                           :func:`approx_set`                            :func:`lag`
    :func:`any_match`                         :func:`day_of_week`                       :func:`jaccard_index`                     :func:`reidentification_potential`        :func:`st_point`                              :func:`arbitrary`                             :func:`last_value`
    :func:`any_values_match`                  :func:`day_of_year`                       :func:`json_array_contains`               :func:`remove_nulls`                      :func:`st_pointn`                             :func:`array_agg`                             :func:`lead`
    :func:`array_average`                     :func:`degrees`                           :func:`json_array_get`                    render                                    :func:`st_points`                             :func:`avg`                                   :func:`nth_value`
    :func:`array_cum_sum`                     :func:`dot_product`                       :func:`json_array_length`                 :func:`repeat`                            :func:`st_polygon`                            :func:`bitwise_and_agg`                       :func:`ntile`
    :func:`array_distinct`                    :func:`dow`                               :func:`json_extract`                      :func:`replace`                           :func:`st_relate`                             :func:`bitwise_or_agg`                        :func:`percent_rank`
    :func:`array_duplicates`                  :func:`doy`                               :func:`json_extract_scalar`               :func:`replace_first`                     :func:`st_startpoint`                         :func:`bool_and`                              :func:`rank`
    :func:`array_except`                      :func:`e`                                 :func:`json_format`                       :func:`reverse`                           :func:`st_symdifference`                      :func:`bool_or`                               :func:`row_number`
    :func:`array_frequency`                   :func:`element_at`                        :func:`json_parse`                        rgb                                       :func:`st_touches`                            :func:`checksum`
    :func:`array_has_duplicates`              :func:`empty_approx_set`                  :func:`json_size`                         :func:`round`                             :func:`st_union`                              :func:`classification_fall_out`
    :func:`array_intersect`                   :func:`ends_with`                         :func:`key_sampling_percent`              :func:`rpad`                              :func:`st_within`                             :func:`classification_miss_rate`
    :func:`array_join`                        :func:`enum_key`                          l2_squared                                :func:`rtrim`                             :func:`st_x`                                  :func:`classification_precision`
    array_least_frequent                      :func:`exp`                               :func:`laplace_cdf`                       :func:`s2_cell_area_sq_km`                :func:`st_xmax`                               :func:`classification_recall`
    :func:`array_max`                         :func:`expand_envelope`                   :func:`last_day_of_month`                 :func:`s2_cell_contains`                  :func:`st_xmin`                               :func:`classification_thresholds`
    :func:`array_max_by`                      :func:`f_cdf`                             :func:`least`                             :func:`s2_cell_from_token`                :func:`st_y`                                  :func:`convex_hull_agg`
    :func:`array_min`                         features                                  :func:`length`                            :func:`s2_cell_level`                     :func:`st_ymax`                               :func:`corr`
    :func:`array_min_by`                      :func:`filter`                            :func:`levenshtein_distance`              :func:`s2_cell_parent`                    :func:`st_ymin`                               :func:`count`
    :func:`array_normalize`                   :func:`find_first`                        :func:`line_interpolate_point`            :func:`s2_cell_to_token`                  :func:`starts_with`                           :func:`count_if`
    :func:`array_position`                    :func:`find_first_index`                  :func:`line_locate_point`                 :func:`s2_cells`                          :func:`strpos`                                :func:`covar_pop`
    :func:`array_remove`                      :func:`flatten`                           :func:`ln`                                :func:`scale_qdigest`                     :func:`strrpos`                               :func:`covar_samp`
    :func:`array_sort`                        :func:`flatten_geometry_collections`      :func:`localtime`                         :func:`second`                            :func:`substr`                                differential_entropy
    :func:`array_sort_desc`                   :func:`floor`                             :func:`localtimestamp`                    :func:`secure_rand`                       :func:`tan`                                   :func:`entropy`
    :func:`array_split_into_chunks`           :func:`fnv1_32`                           :func:`log10`                             :func:`secure_random`                     :func:`tanh`                                  evaluate_classifier_predictions
    :func:`array_sum`                         :func:`fnv1_64`                           :func:`log2`                              :func:`sequence`                          tdigest_agg                                   :func:`every`
    :func:`array_top_n`                       :func:`fnv1a_32`                          :func:`longest_common_prefix`             :func:`sha1`                              :func:`timezone_hour`                         :func:`geometric_mean`
    :func:`array_union`                       :func:`fnv1a_64`                          :func:`lower`                             :func:`sha256`                            :func:`timezone_minute`                       :func:`geometry_union_agg`
    :func:`arrays_overlap`                    :func:`format_datetime`                   :func:`lpad`                              :func:`sha512`                            :func:`to_base`                               :func:`histogram`
    :func:`asin`                              :func:`from_base`                         :func:`ltrim`                             :func:`shuffle`                           to_base32                                     :func:`khyperloglog_agg`
    :func:`atan`                              :func:`from_base32`                       :func:`map`                               :func:`sign`                              :func:`to_base64`                             :func:`kurtosis`
    :func:`atan2`                             :func:`from_base64`                       :func:`map_concat`                        :func:`simplify_geometry`                 :func:`to_base64url`                          learn_classifier
    bar                                       :func:`from_base64url`                    :func:`map_entries`                       :func:`sin`                               :func:`to_big_endian_32`                      learn_libsvm_classifier
    :func:`beta_cdf`                          :func:`from_big_endian_32`                :func:`map_filter`                        sketch_kll_quantile                       :func:`to_big_endian_64`                      learn_libsvm_regressor
    :func:`bing_tile`                         :func:`from_big_endian_64`                :func:`map_from_entries`                  sketch_kll_rank                           :func:`to_geometry`                           learn_regressor
    :func:`bing_tile_at`                      :func:`from_hex`                          :func:`map_keys`                          :func:`slice`                             :func:`to_hex`                                :func:`make_set_digest`
    :func:`bing_tile_children`                :func:`from_ieee754_32`                   :func:`map_keys_by_top_n_values`          spatial_partitions                        :func:`to_ieee754_32`                         :func:`map_agg`
    :func:`bing_tile_coordinates`             :func:`from_ieee754_64`                   :func:`map_normalize`                     :func:`split`                             :func:`to_ieee754_64`                         :func:`map_union`
    :func:`bing_tile_parent`                  :func:`from_iso8601_date`                 :func:`map_remove_null_values`            :func:`split_part`                        :func:`to_iso8601`                            :func:`map_union_sum`
    :func:`bing_tile_polygon`                 :func:`from_iso8601_timestamp`            :func:`map_subset`                        :func:`split_to_map`                      :func:`to_milliseconds`                       :func:`max`
    :func:`bing_tile_quadkey`                 :func:`from_unixtime`                     :func:`map_top_n`                         :func:`split_to_multimap`                 :func:`to_spherical_geography`                :func:`max_by`
    :func:`bing_tile_zoom_level`              :func:`from_utf8`                         :func:`map_top_n_keys`                    :func:`spooky_hash_v2_32`                 :func:`to_unixtime`                           :func:`merge`
    :func:`bing_tiles_around`                 :func:`gamma_cdf`                         map_top_n_keys_by_value                   :func:`spooky_hash_v2_64`                 :func:`to_utf8`                               :func:`merge_set_digest`
    :func:`binomial_cdf`                      :func:`geometry_as_geojson`               :func:`map_top_n_values`                  :func:`sqrt`                              :func:`trail`                                 :func:`min`
    :func:`bit_count`                         :func:`geometry_from_geojson`             :func:`map_values`                        :func:`st_area`                           :func:`transform`                             :func:`min_by`
    :func:`bit_length`                        :func:`geometry_invalid_reason`           :func:`map_zip_with`                      :func:`st_asbinary`                       :func:`transform_keys`                        :func:`multimap_agg`
    :func:`bitwise_and`                       :func:`geometry_nearest_points`           :func:`md5`                               :func:`st_astext`                         :func:`transform_values`                      :func:`noisy_avg_gaussian`
    :func:`bitwise_arithmetic_shift_right`    :func:`geometry_to_bing_tiles`            :func:`merge_hll`                         :func:`st_boundary`                       :func:`trim`                                  :func:`noisy_count_gaussian`
    :func:`bitwise_left_shift`                :func:`geometry_to_dissolved_bing_tiles`  :func:`merge_khll`                        :func:`st_buffer`                         :func:`trim_array`                            :func:`noisy_count_if_gaussian`
    :func:`bitwise_logical_shift_right`       :func:`geometry_union`                    :func:`millisecond`                       :func:`st_centroid`                       :func:`truncate`                              :func:`noisy_sum_gaussian`
    :func:`bitwise_not`                       :func:`google_polyline_decode`            :func:`minute`                            :func:`st_contains`                       :func:`typeof`                                :func:`numeric_histogram`
    :func:`bitwise_or`                        :func:`google_polyline_encode`            :func:`mod`                               :func:`st_convexhull`                     :func:`uniqueness_distribution`               :func:`qdigest_agg`
    :func:`bitwise_right_shift`               :func:`great_circle_distance`             :func:`month`                             :func:`st_coorddim`                       :func:`upper`                                 :func:`reduce_agg`
    :func:`bitwise_right_shift_arithmetic`    :func:`greatest`                          :func:`multimap_from_entries`             :func:`st_crosses`                        :func:`url_decode`                            :func:`regr_avgx`
    :func:`bitwise_shift_left`                :func:`hamming_distance`                  :func:`murmur3_x64_128`                   :func:`st_difference`                     :func:`url_encode`                            :func:`regr_avgy`
    :func:`bitwise_xor`                       :func:`hash_counts`                       myanmar_font_encoding                     :func:`st_dimension`                      :func:`url_extract_fragment`                  :func:`regr_count`
    :func:`cardinality`                       :func:`hmac_md5`                          myanmar_normalize_unicode                 :func:`st_disjoint`                       :func:`url_extract_host`                      :func:`regr_intercept`
    :func:`cauchy_cdf`                        :func:`hmac_sha1`                         :func:`nan`                               :func:`st_distance`                       :func:`url_extract_parameter`                 :func:`regr_r2`
    :func:`cbrt`                              :func:`hmac_sha256`                       :func:`ngrams`                            :func:`st_endpoint`                       :func:`url_extract_path`                      :func:`regr_slope`
    :func:`ceil`                              :func:`hmac_sha512`                       :func:`no_keys_match`                     :func:`st_envelope`                       :func:`url_extract_port`                      :func:`regr_sxx`
    :func:`ceiling`                           :func:`hour`                              :func:`no_values_match`                   :func:`st_envelopeaspts`                  :func:`url_extract_protocol`                  :func:`regr_sxy`
    :func:`chi_squared_cdf`                   :func:`infinity`                          :func:`none_match`                        :func:`st_equals`                         :func:`url_extract_query`                     :func:`regr_syy`
    :func:`chr`                               :func:`intersection_cardinality`          :func:`normal_cdf`                        :func:`st_exteriorring`                   :func:`uuid`                                  :func:`reservoir_sample`
    classify                                  :func:`inverse_beta_cdf`                  :func:`normalize`                         :func:`st_geometries`                     :func:`value_at_quantile`                     :func:`set_agg`
    :func:`codepoint`                         :func:`inverse_binomial_cdf`              :func:`now`                               :func:`st_geometryfromtext`               :func:`values_at_quantiles`                   :func:`set_union`
    color                                     :func:`inverse_cauchy_cdf`                :func:`parse_datetime`                    :func:`st_geometryn`                      :func:`week`                                  sketch_kll
    :func:`combinations`                      :func:`inverse_chi_squared_cdf`           :func:`parse_duration`                    :func:`st_geometrytype`                   :func:`week_of_year`                          sketch_kll_with_k
    :func:`concat`                            :func:`inverse_f_cdf`                     :func:`parse_presto_data_size`            :func:`st_geomfrombinary`                 :func:`weibull_cdf`                           :func:`skewness`
    :func:`contains`                          :func:`inverse_gamma_cdf`                 :func:`pi`                                :func:`st_interiorringn`                  :func:`width_bucket`                          spatial_partitioning
    :func:`cos`                               :func:`inverse_laplace_cdf`               pinot_binary_decimal_to_double            :func:`st_interiorrings`                  :func:`wilson_interval_lower`                 :func:`stddev`
    :func:`cosh`                              :func:`inverse_normal_cdf`                :func:`poisson_cdf`                       :func:`st_intersection`                   :func:`wilson_interval_upper`                 :func:`stddev_pop`
    :func:`cosine_similarity`                 :func:`inverse_poisson_cdf`               :func:`pow`                               :func:`st_intersects`                     :func:`word_stem`                             :func:`stddev_samp`
    :func:`crc32`                             :func:`inverse_weibull_cdf`               :func:`power`                             :func:`st_isclosed`                       :func:`xxhash64`                              :func:`sum`
    :func:`current_date`                      :func:`ip_prefix`                         :func:`quantile_at_value`                 :func:`st_isempty`                        :func:`year`                                  :func:`tdigest_agg`
    :func:`current_time`                      :func:`ip_prefix_collapse`                :func:`quarter`                           :func:`st_isring`                         :func:`year_of_week`                          :func:`var_pop`
    :func:`current_timestamp`                 :func:`ip_prefix_subnets`                 :func:`radians`                           :func:`st_issimple`                       :func:`yow`                                   :func:`var_samp`
    :func:`current_timezone`                  :func:`ip_subnet_max`                     :func:`rand`                              :func:`st_isvalid`                        :func:`zip`                                   :func:`variance`
    :func:`date`                              :func:`ip_subnet_min`                     :func:`random`                            :func:`st_length`                         :func:`zip_with`
    :func:`date_add`                          :func:`ip_subnet_range`                   :func:`reduce`                            :func:`st_linefromtext`
    :func:`date_diff`                         :func:`is_finite`                         :func:`regexp_extract`                    :func:`st_linestring`
    :func:`date_format`                       :func:`is_infinite`                       :func:`regexp_extract_all`                :func:`st_multipoint`
    ========================================  ========================================  ========================================  ========================================  ========================================  ==  ========================================  ==  ========================================
