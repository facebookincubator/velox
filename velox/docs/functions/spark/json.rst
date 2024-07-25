==============
JSON Functions
==============

.. spark:function:: json_array_length(jsonString) -> integer

    Returns the number of elements in the outermost JSON array. NULL is returned in case of any other valid JSON string, NULL or an invalid JSON. ::

        SELECT json_array_length('[1,2,3,4]'); -- 4
        SELECT json_array_length('1'); -- NULL
        SELECT json_array_length(NULL); -- NULL
        SELECT json_array_length(1); -- NULL

.. spark:function:: json_object_keys(jsonString) -> array(string)

    Returns all the keys of the outermost JSON object as an array if a valid JSON object is given.  If it is any other valid JSON string, an invalid JSON string or an empty string, the function returns null. ::

        SELECT json_object_keys('{}'); -- []
        SELECT json_object_keys('{"name": "Alice", "age": 5, "id": "001"}'); -- ['name', 'age', 'id']
        SELECT json_object_keys(''); -- NULL
        SELECT json_object_keys(1); -- NULL
        SELECT json_object_keys('"hello"'); -- NULL
