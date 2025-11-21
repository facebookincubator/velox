====================
Spatial Join Fuzzer
====================

Overview
========

The Spatial Join Fuzzer tests the correctness of the SpatialJoin operator by generating random geometry data and spatial join plans. It verifies that SpatialJoin produces the same results as NestedLoopJoin for equivalent queries.


Supported Features
==================

Join Types
----------

The fuzzer tests the two join types supported by SpatialJoin (as defined in ``SpatialJoinNode::isSupported()``):

* **INNER** - Only matching rows from both sides
* **LEFT** - All rows from left side, matched rows from right side

Spatial Predicates
------------------

The fuzzer tests these spatial predicates:

* ``ST_Intersects(geometry1, geometry2)`` - Tests if geometries intersect
* ``ST_Contains(geometry1, geometry2)`` - Tests if one geometry contains another
* ``ST_Within(geometry1, geometry2)`` - Tests if one geometry is within another
* ``ST_Distance(geometry1, geometry2) < threshold`` - Tests distance with threshold

Geometry Types
--------------

The fuzzer generates Well-Known Text (WKT) strings for three geometry types:

* **POINT** - Single coordinate point (e.g., ``POINT (10.5 20.3)``)
* **POLYGON** - Closed shape with vertices
* **LINESTRING** - Line segment between two points

Distribution Patterns
---------------------

Geometries are generated using three distribution patterns:

* **Uniform** - Geometries uniformly distributed in space (0-1000 range)
* **Clustered** - Geometries grouped in 5 specific regions to test overlap scenarios
* **Sparse** - Geometries widely spread (0-2000 range) with low overlap probability

Implementation Details
======================


Geometry Generation
-------------------

Geometries are generated using ``AbstractInputGenerator`` subclasses:

* ``PointInputGenerator`` - Generates POINT WKT strings
* ``PolygonInputGenerator`` - Generates POLYGON WKT strings
* ``LineStringInputGenerator`` - Generates LINESTRING WKT strings

Each generator implements the ``generate(vector_size_t index)`` method to produce geometry strings based on the distribution pattern.

**Uniform Distribution**::

    x = random(0, 1000)
    y = random(0, 1000)
    POINT (x y)

**Clustered Distribution**::

    cluster = row % 5  // 5 clusters
    centerX = cluster * 200 + 100
    centerY = cluster * 200 + 100
    x = centerX + random(-50, 50)
    y = centerY + random(-50, 50)
    POINT (x y)

**Sparse Distribution**::

    x = random(0, 2000)  // Larger Range
    y = random(0, 2000)
    POINT (x y)

Data Matching Strategy
----------------------

To ensure some matches occur during joins:

* Build side copies ~30% of geometries from probe side
* 10% chance of empty build side to test edge cases

Verification
------------

The fuzzer compares results from two equivalent plans:

1. **SpatialJoin plan** - Using the specialized SpatialJoin operator
2. **NestedLoopJoin plan** - Using NestedLoopJoin with the same spatial predicate as a filter

Results must match exactly, validating that SpatialJoin implements spatial predicates correctly.

Key Differences from JoinFuzzer
================================

Join Conditions
---------------

Unlike regular joins with simple equality predicates::

    // Regular join
    probe.id = build.id

    // Spatial join
    ST_Intersects(probe_geom, build_geom)

Spatial joins use **function call expressions** as join conditions rather than simple column references.
