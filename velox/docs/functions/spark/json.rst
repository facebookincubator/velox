==============
JSON Functions
==============

JSON Format
-----------

JSON is a language-independent data format that represents data as
human-readable text. A JSON text can represent a number, a boolean, a
string, an array, an object, or a null. A JSON text representing a string
must escape all characters and enclose the string in double quotes, e.g.,
``"123\n"``, whereas a JSON text representing a number does not need to,
e.g., ``123``. A JSON text representing an array must enclose the array
elements in square brackets, e.g., ``[1,2,3]``. More detailed grammar can
be found in `this JSON introduction`_.

.. _this JSON introduction: https://www.json.org

JSON Functions
--------------

.. spark:function:: get_json_object(jsonString, path) -> varchar

    Returns a json object, represented by VARCHAR, from ``jsonString`` by searching ``path``.
    Valid ``path`` should start with '$' and then contain "[index]", "['field']" or ".field"
    to define a JSON path. Here are some examples: "$.a" "$.a.b", "$[0]['a'].b". Returns
    ``jsonString`` if ``path`` is "$". Returns NULL if ``jsonString`` or ``path`` is malformed.
    Returns NULL if ``path`` does not exist. ::

        SELECT get_json_object('{"a":"b"}', '$.a'); -- 'b'
        SELECT get_json_object('{"a":{"b":"c"}}', '$.a'); -- '{"b":"c"}'
        SELECT get_json_object('{"a":3}', '$.b'); -- NULL (unexisting field)
        SELECT get_json_object('{"a"-3}'', '$.a'); -- NULL (malformed JSON string)
        SELECT get_json_object('{"a":3}'', '.a'); -- NULL (malformed JSON path)

.. spark:function:: json_object_keys(jsonString) -> array(string)

    Returns all the keys of the outermost JSON object as an array if a valid JSON object is given.
    If it is any other valid JSON string, an invalid JSON string or an empty string, the function
    returns null. ::

        SELECT json_object_keys('{}'); -- []
        SELECT json_object_keys('{"name": "Alice", "age": 5, "id": "001"}'); -- ['name', 'age', 'id']
        SELECT json_object_keys(''); -- NULL
        SELECT json_object_keys(1); -- NULL
        SELECT json_object_keys('"hello"'); -- NULL

.. spark:function:: from_json(jsonString) -> [json object]

    Casts a JSON string to an ARRAY, MAP, or ROW type, with the output type 
    determined by the expression. Returns NULL, if the input string is unparsable.
    Supported element types include BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, 
    REAL, DOUBLE, VARCHAR, ARRAY, MAP, and ROW. When casting to ARRAY or MAP, 
    the element type of the array or the value type of the map must be one of 
    these supported types, and for maps, the key type must be VARCHAR. Casting 
    to ROW supports only JSON objects, where the keys must exactly match the ROW 
    field names (case sensitivity).
    Behaviors of the casts are shown with the examples below. ::

        SELECT from_json('{"a": true}'); -- {'a'=true} // Output type: ROW({"a"}, {BOOLEAN()})
        SELECT from_json('{"a": 1}'); -- {'a'=1} // Output type: ROW({"a"}, {INTEGER()})
        SELECT from_json('{"a": 1.0}'); -- {'a'=1.0} // Output type: ROW({"a"}, {DOUBLE()})
        SELECT from_json('["name", "age", "id"]'); -- ['name', 'age', 'id'] // Output type: ARRAY(VARCHAR())
        SELECT from_json('{"a": 1, "b": 2}'); -- {'a'=1, 'b'=2} // Output type: MAP(VARCHAR(),INTEGER())
