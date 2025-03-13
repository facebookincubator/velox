====================
Geospatial Operators
====================

Bing Tile Functions
-------------------

Bing tiles are a convenient quad-tree representation of the WGS84 projection of
Earth's surface.  They can be used to partition geospatial data, perform quick
proximity or intersection checks, and more.  Each tile is defined by a `zoom`
level (how far down the quad-tree the tile lives), and an `x` and `y` coordinate
specifying where it is in that `zoom` level.  Velox supports `zoom` levels from
0 to 23.  For a given zoom level, `x` and `y` must be between 0 and `2**zoom -
1` inclusive.  Lower `x` values are west of higher `x` values, and lower `y`
values are north of higher `y` values.

Bing tiles can be cast to and from an efficient BIGINT representation. While every
Bing tile can be cast to a valid BIGINT, not every BIGINT is a valid Bing tile, so
casting BIGINT to BINGTILE may fail.

::
    CAST(CAST(tile AS BIGINT) AS BINGTILE)

See https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
for more details.

.. function:: bing_tile(x: integer, y: integer, zoom_level: tinyint) -> tile: BingTile

    Creates a Bing tile object from `x`, `y` coordinates and a `zoom_level`.
    Zoom levels from 0 to 23 are supported, with valid `x` and `y` coordinates
    described above.  Invalid parameters will return a User Error.

.. function:: bing_tile_coordinates(tile: BingTile) -> coords: row(integer,integer)

   Returns the `x`, `y` coordinates of a given Bing tile as `row(x, y)`.

.. function:: bing_tile_zoom_level(tile: BingTile) -> zoom_level: tinyint

   Returns the zoom level of a given Bing tile.
