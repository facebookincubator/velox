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
    table.coverage tr:nth-child(4) td:nth-child(4) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(7) td:nth-child(4) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(12) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(12) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(12) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(13) td:nth-child(3) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(15) td:nth-child(3) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(18) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(18) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(19) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(19) td:nth-child(2) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(26) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(26) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(26) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(26) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(27) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(28) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(29) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(29) td:nth-child(2) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(31) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(31) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(32) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(33) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(34) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(34) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(34) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(35) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(35) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(35) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(35) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(36) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(37) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(38) td:nth-child(4) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(44) td:nth-child(3) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(54) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(54) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(55) td:nth-child(1) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(57) td:nth-child(3) {background-color: #6BA81E;}
    table.coverage tr:nth-child(57) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(57) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(57) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(58) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(58) td:nth-child(2) {background-color: #6BA81E;}
    table.coverage tr:nth-child(58) td:nth-child(3) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(61) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(61) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(61) td:nth-child(7) {background-color: #6BA81E;}
    table.coverage tr:nth-child(62) td:nth-child(1) {background-color: #6BA81E;}
    table.coverage tr:nth-child(62) td:nth-child(2) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(71) td:nth-child(3) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(75) td:nth-child(4) {background-color: #6BA81E;}
    table.coverage tr:nth-child(75) td:nth-child(5) {background-color: #6BA81E;}
    table.coverage tr:nth-child(75) td:nth-child(7) {background-color: #6BA81E;}
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
    table.coverage tr:nth-child(80) td:nth-child(5) {background-color: #6BA81E;}
    </style>

.. table::
    :widths: auto
    :class: coverage

    ========================================  ========================================  ========================================  ========================================  ========================================  ==  ========================================  ==  ========================================
    Scalar Functions                                                                                                                                                                                                      Aggregate Functions                           Window Functions
    ================================================================================================================================================================================================================  ==  ========================================  ==  ========================================
    :func:`abs`                               :func:`date_diff`                         :func:`ip_subnet_min`                     :func:`radians`                           :func:`st_multipoint`                         :func:`approx_distinct`                       :func:`cume_dist`
    :func:`acos`                              :func:`date_format`                       :func:`ip_subnet_range`                   :func:`rand`                              :func:`st_numgeometries`                      :func:`approx_most_frequent`                  :func:`dense_rank`
    :func:`all_match`                         :func:`date_parse`                        :func:`is_finite`                         :func:`random`                            :func:`st_numinteriorring`                    :func:`approx_percentile`                     :func:`first_value`
    :func:`any_keys_match`                    :func:`date_trunc`                        :func:`is_infinite`                       :func:`reduce`                            :func:`st_numpoints`                          :func:`approx_set`                            :func:`lag`
    :func:`any_match`                         :func:`day`                               :func:`is_json_scalar`                    :func:`regexp_extract`                    :func:`st_overlaps`                           :func:`arbitrary`                             :func:`last_value`
    :func:`any_values_match`                  :func:`day_of_month`                      :func:`is_nan`                            :func:`regexp_extract_all`                :func:`st_point`                              :func:`array_agg`                             :func:`lead`
    :func:`array_average`                     :func:`day_of_week`                       :func:`is_private_ip`                     :func:`regexp_like`                       :func:`st_pointn`                             :func:`avg`                                   :func:`nth_value`
    :func:`array_cum_sum`                     :func:`day_of_year`                       :func:`is_subnet_of`                      :func:`regexp_replace`                    :func:`st_points`                             :func:`bitwise_and_agg`                       :func:`ntile`
    :func:`array_distinct`                    :func:`degrees`                           :func:`jaccard_index`                     :func:`regexp_split`                      :func:`st_polygon`                            :func:`bitwise_or_agg`                        :func:`percent_rank`
    :func:`array_duplicates`                  :func:`dot_product`                       :func:`json_array_contains`               regress                                   :func:`st_relate`                             :func:`bool_and`                              :func:`rank`
    :func:`array_except`                      :func:`dow`                               :func:`json_array_get`                    :func:`reidentification_potential`        :func:`st_startpoint`                         :func:`bool_or`                               :func:`row_number`
    :func:`array_frequency`                   :func:`doy`                               :func:`json_array_length`                 :func:`remove_nulls`                      :func:`st_symdifference`                      :func:`checksum`
    :func:`array_has_duplicates`              :func:`e`                                 :func:`json_extract`                      render                                    :func:`st_touches`                            :func:`classification_fall_out`
    :func:`array_intersect`                   :func:`element_at`                        :func:`json_extract_scalar`               :func:`repeat`                            :func:`st_union`                              :func:`classification_miss_rate`
    :func:`array_join`                        :func:`empty_approx_set`                  :func:`json_format`                       :func:`replace`                           :func:`st_within`                             :func:`classification_precision`
    array_least_frequent                      :func:`ends_with`                         :func:`json_parse`                        :func:`replace_first`                     :func:`st_x`                                  :func:`classification_recall`
    :func:`array_max`                         :func:`enum_key`                          :func:`json_size`                         :func:`reverse`                           :func:`st_xmax`                               :func:`classification_thresholds`
    :func:`array_max_by`                      :func:`exp`                               :func:`key_sampling_percent`              rgb                                       :func:`st_xmin`                               :func:`convex_hull_agg`
    :func:`array_min`                         :func:`expand_envelope`                   l2_squared                                :func:`round`                             :func:`st_y`                                  :func:`corr`
    :func:`array_min_by`                      :func:`f_cdf`                             :func:`laplace_cdf`                       :func:`rpad`                              :func:`st_ymax`                               :func:`count`
    :func:`array_normalize`                   features                                  :func:`last_day_of_month`                 :func:`rtrim`                             :func:`st_ymin`                               :func:`count_if`
    :func:`array_position`                    :func:`filter`                            :func:`least`                             :func:`scale_qdigest`                     :func:`starts_with`                           :func:`covar_pop`
    :func:`array_remove`                      :func:`find_first`                        :func:`length`                            :func:`second`                            :func:`strpos`                                :func:`covar_samp`
    :func:`array_sort`                        :func:`find_first_index`                  :func:`levenshtein_distance`              :func:`secure_rand`                       :func:`strrpos`                               differential_entropy
    :func:`array_sort_desc`                   :func:`flatten`                           :func:`line_interpolate_point`            :func:`secure_random`                     :func:`substr`                                :func:`entropy`
    array_split_into_chunks                   :func:`flatten_geometry_collections`      :func:`line_locate_point`                 :func:`sequence`                          :func:`tan`                                   evaluate_classifier_predictions
    :func:`array_sum`                         :func:`floor`                             :func:`ln`                                :func:`sha1`                              :func:`tanh`                                  :func:`every`
    :func:`array_top_n`                       :func:`fnv1_32`                           :func:`localtime`                         :func:`sha256`                            tdigest_agg                                   :func:`geometric_mean`
    :func:`array_union`                       :func:`fnv1_64`                           localtimestamp                            :func:`sha512`                            :func:`timezone_hour`                         :func:`geometry_union_agg`
    :func:`arrays_overlap`                    :func:`fnv1a_32`                          :func:`log10`                             :func:`shuffle`                           :func:`timezone_minute`                       :func:`histogram`
    :func:`asin`                              :func:`fnv1a_64`                          :func:`log2`                              :func:`sign`                              :func:`to_base`                               :func:`khyperloglog_agg`
    :func:`atan`                              :func:`format_datetime`                   :func:`longest_common_prefix`             :func:`simplify_geometry`                 to_base32                                     :func:`kurtosis`
    :func:`atan2`                             :func:`from_base`                         :func:`lower`                             :func:`sin`                               :func:`to_base64`                             learn_classifier
    bar                                       :func:`from_base32`                       :func:`lpad`                              sketch_kll_quantile                       :func:`to_base64url`                          learn_libsvm_classifier
    :func:`beta_cdf`                          :func:`from_base64`                       :func:`ltrim`                             sketch_kll_rank                           :func:`to_big_endian_32`                      learn_libsvm_regressor
    :func:`bing_tile`                         :func:`from_base64url`                    :func:`map`                               :func:`slice`                             :func:`to_big_endian_64`                      learn_regressor
    :func:`bing_tile_at`                      :func:`from_big_endian_32`                :func:`map_concat`                        spatial_partitions                        :func:`to_geometry`                           :func:`make_set_digest`
    :func:`bing_tile_children`                :func:`from_big_endian_64`                :func:`map_entries`                       :func:`split`                             :func:`to_hex`                                :func:`map_agg`
    :func:`bing_tile_coordinates`             :func:`from_hex`                          :func:`map_filter`                        :func:`split_part`                        :func:`to_ieee754_32`                         :func:`map_union`
    :func:`bing_tile_parent`                  :func:`from_ieee754_32`                   :func:`map_from_entries`                  :func:`split_to_map`                      :func:`to_ieee754_64`                         :func:`map_union_sum`
    :func:`bing_tile_polygon`                 :func:`from_ieee754_64`                   :func:`map_keys`                          :func:`split_to_multimap`                 :func:`to_iso8601`                            :func:`max`
    :func:`bing_tile_quadkey`                 :func:`from_iso8601_date`                 :func:`map_keys_by_top_n_values`          :func:`spooky_hash_v2_32`                 :func:`to_milliseconds`                       :func:`max_by`
    :func:`bing_tile_zoom_level`              :func:`from_iso8601_timestamp`            :func:`map_normalize`                     :func:`spooky_hash_v2_64`                 :func:`to_spherical_geography`                :func:`merge`
    :func:`bing_tiles_around`                 :func:`from_unixtime`                     :func:`map_remove_null_values`            :func:`sqrt`                              :func:`to_unixtime`                           :func:`merge_set_digest`
    :func:`binomial_cdf`                      :func:`from_utf8`                         :func:`map_subset`                        :func:`st_area`                           :func:`to_utf8`                               :func:`min`
    :func:`bit_count`                         :func:`gamma_cdf`                         :func:`map_top_n`                         :func:`st_asbinary`                       :func:`trail`                                 :func:`min_by`
    :func:`bit_length`                        :func:`geometry_as_geojson`               :func:`map_top_n_keys`                    :func:`st_astext`                         :func:`transform`                             :func:`multimap_agg`
    :func:`bitwise_and`                       :func:`geometry_from_geojson`             map_top_n_keys_by_value                   :func:`st_boundary`                       :func:`transform_keys`                        :func:`noisy_avg_gaussian`
    :func:`bitwise_arithmetic_shift_right`    :func:`geometry_invalid_reason`           :func:`map_top_n_values`                  :func:`st_buffer`                         :func:`transform_values`                      :func:`noisy_count_gaussian`
    :func:`bitwise_left_shift`                :func:`geometry_nearest_points`           :func:`map_values`                        :func:`st_centroid`                       :func:`trim`                                  :func:`noisy_count_if_gaussian`
    :func:`bitwise_logical_shift_right`       :func:`geometry_to_bing_tiles`            :func:`map_zip_with`                      :func:`st_contains`                       :func:`trim_array`                            :func:`noisy_sum_gaussian`
    :func:`bitwise_not`                       :func:`geometry_to_dissolved_bing_tiles`  :func:`md5`                               :func:`st_convexhull`                     :func:`truncate`                              :func:`numeric_histogram`
    :func:`bitwise_or`                        :func:`geometry_union`                    :func:`merge_hll`                         :func:`st_coorddim`                       :func:`typeof`                                :func:`qdigest_agg`
    :func:`bitwise_right_shift`               google_polyline_decode                    :func:`merge_khll`                        :func:`st_crosses`                        :func:`uniqueness_distribution`               :func:`reduce_agg`
    :func:`bitwise_right_shift_arithmetic`    google_polyline_encode                    :func:`millisecond`                       :func:`st_difference`                     :func:`upper`                                 :func:`regr_avgx`
    :func:`bitwise_shift_left`                :func:`great_circle_distance`             :func:`minute`                            :func:`st_dimension`                      :func:`url_decode`                            :func:`regr_avgy`
    :func:`bitwise_xor`                       :func:`greatest`                          :func:`mod`                               :func:`st_disjoint`                       :func:`url_encode`                            :func:`regr_count`
    :func:`cardinality`                       :func:`hamming_distance`                  :func:`month`                             :func:`st_distance`                       :func:`url_extract_fragment`                  :func:`regr_intercept`
    :func:`cauchy_cdf`                        :func:`hash_counts`                       :func:`multimap_from_entries`             :func:`st_endpoint`                       :func:`url_extract_host`                      :func:`regr_r2`
    :func:`cbrt`                              :func:`hmac_md5`                          :func:`murmur3_x64_128`                   :func:`st_envelope`                       :func:`url_extract_parameter`                 :func:`regr_slope`
    :func:`ceil`                              :func:`hmac_sha1`                         myanmar_font_encoding                     :func:`st_envelopeaspts`                  :func:`url_extract_path`                      :func:`regr_sxx`
    :func:`ceiling`                           :func:`hmac_sha256`                       myanmar_normalize_unicode                 :func:`st_equals`                         :func:`url_extract_port`                      :func:`regr_sxy`
    :func:`chi_squared_cdf`                   :func:`hmac_sha512`                       :func:`nan`                               :func:`st_exteriorring`                   :func:`url_extract_protocol`                  :func:`regr_syy`
    :func:`chr`                               :func:`hour`                              :func:`ngrams`                            :func:`st_geometries`                     :func:`url_extract_query`                     :func:`reservoir_sample`
    classify                                  :func:`infinity`                          :func:`no_keys_match`                     :func:`st_geometryfromtext`               :func:`uuid`                                  :func:`set_agg`
    :func:`codepoint`                         :func:`intersection_cardinality`          :func:`no_values_match`                   :func:`st_geometryn`                      :func:`value_at_quantile`                     :func:`set_union`
    color                                     :func:`inverse_beta_cdf`                  :func:`none_match`                        :func:`st_geometrytype`                   :func:`values_at_quantiles`                   sketch_kll
    :func:`combinations`                      :func:`inverse_binomial_cdf`              :func:`normal_cdf`                        :func:`st_geomfrombinary`                 :func:`week`                                  sketch_kll_with_k
    :func:`concat`                            :func:`inverse_cauchy_cdf`                :func:`normalize`                         :func:`st_interiorringn`                  :func:`week_of_year`                          :func:`skewness`
    :func:`contains`                          :func:`inverse_chi_squared_cdf`           :func:`now`                               :func:`st_interiorrings`                  :func:`weibull_cdf`                           spatial_partitioning
    :func:`cos`                               :func:`inverse_f_cdf`                     :func:`parse_datetime`                    :func:`st_intersection`                   :func:`width_bucket`                          :func:`stddev`
    :func:`cosh`                              :func:`inverse_gamma_cdf`                 :func:`parse_duration`                    :func:`st_intersects`                     :func:`wilson_interval_lower`                 :func:`stddev_pop`
    :func:`cosine_similarity`                 :func:`inverse_laplace_cdf`               :func:`parse_presto_data_size`            :func:`st_isclosed`                       :func:`wilson_interval_upper`                 :func:`stddev_samp`
    :func:`crc32`                             :func:`inverse_normal_cdf`                :func:`pi`                                :func:`st_isempty`                        :func:`word_stem`                             :func:`sum`
    :func:`current_date`                      :func:`inverse_poisson_cdf`               pinot_binary_decimal_to_double            :func:`st_isring`                         :func:`xxhash64`                              :func:`tdigest_agg`
    current_time                              :func:`inverse_weibull_cdf`               :func:`poisson_cdf`                       :func:`st_issimple`                       :func:`year`                                  :func:`var_pop`
    :func:`current_timestamp`                 :func:`ip_prefix`                         :func:`pow`                               :func:`st_isvalid`                        :func:`year_of_week`                          :func:`var_samp`
    :func:`current_timezone`                  :func:`ip_prefix_collapse`                :func:`power`                             :func:`st_length`                         :func:`yow`                                   :func:`variance`
    :func:`date`                              :func:`ip_prefix_subnets`                 :func:`quantile_at_value`                 :func:`st_linefromtext`                   :func:`zip`
    :func:`date_add`                          :func:`ip_subnet_max`                     :func:`quarter`                           :func:`st_linestring`                     :func:`zip_with`
    ========================================  ========================================  ========================================  ========================================  ========================================  ==  ========================================  ==  ========================================
