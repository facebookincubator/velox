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

.. spark:function:: to_json(jsonObject) -> jsonString

    Converts a Json object (ROW, ARRAY or MAP) into a JSON string.
    The current implementation has following limitations.

    * Does not support user provided options. ::

        to_json(ROW(1, "a"), map('option', 'value'))

    Examples of valid inputs are listed as below. ::

        SELECT to_json(ROW(1, "a")); -- {"a":1}
        SELECT to_json(ARRAY[1, 2, 3]); -- [1,2,3]
        SELECT to_json(MAP(ARRAY['x', 'y'], ARRAY[1, 2])); -- {"x":1,"y":2}
