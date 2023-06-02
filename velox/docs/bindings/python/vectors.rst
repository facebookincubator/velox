################
PyVelox: Vectors
################

This is a quick introduction to Velox vectors in Python. 

Check out :doc:`Vectors <../../develop/vectors>` guide to learn more about
Velox vectors.

Let's take a look at a few ways we can create vectors.

Creating Vectors
-----------------

Flat Vectors - Scalar Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating :ref:`developer-flat-vector-scalar-types` using a python list. 

.. doctest::

    >>> import pyvelox.pyvelox as pv
    >>> flat_vec = pv.from_list([1, 2, 3])
    >>> print(flat_vec)
    [FLAT BIGINT: 3 elements, no nulls]
    0: 1
    1: 2
    2: 3


Constant Vectors - Scalar Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating :ref:`developer-constant-vector-scalar-types` using a python list. 


.. doctest::

    >>> import pyvelox.pyvelox as pv
    >>> const_vec =  pv.constant_vector(10, 3)
    >>> print(const_vec)
    [CONSTANT BIGINT: 3 elements, 10]
    0: 10
    1: 10
    2: 10


Dictionary Vector - Scalar Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating :ref:`developer-dictionary-vector-scalar-types` using a python list. 

.. doctest::

    >>> import pyvelox.pyvelox as pv
    >>> base_indices = [0, 0, 1, 0, 2]
    >>> dict_vec = pv.dictionary_vector(pv.from_list([1, 2, 3]), base_indices)
    >>> print(dict_vec)
    [DICTIONARY BIGINT: 5 elements, no nulls]
    0: [0->0] 1
    1: [1->0] 1
    2: [2->1] 2
    3: [3->0] 1
    4: [4->2] 3


Indices
'''''''

.. doctest::

    >>> import pyvelox.pyvelox as pv
    >>> base_indices = [0, 0, 1, 0, 2]
    >>> dict_vec = pv.dictionary_vector(pv.from_list([1, 2, 3]), base_indices)
    >>> indices = dict_vec.indices()
    >>> list(indices)
    [0, 0, 1, 0, 2]



Vector Encodings
----------------

Flat Vector
^^^^^^^^^^^

.. doctest::

    >>> import pyvelox.pyvelox as pv
    >>> int_vec =  pv.from_list([1, 2, 3])
    >>> int_vec.encoding()
    <VectorEncodingSimple.FLAT: 3>
    >>> float_vec =  pv.from_list([1.0, 2.0, 3.0])
    >>> float_vec.encoding()
    <VectorEncodingSimple.FLAT: 3>

Constant Vector
^^^^^^^^^^^^^^^

.. doctest::

    >>> import pyvelox.pyvelox as pv
    >>> ints = pv.constant_vector(10, 3)
    >>> ints.encoding()
    <VectorEncodingSimple.CONSTANT: 1>


Dictionary Vector
^^^^^^^^^^^^^^^

Doesn't support this functionality.


Vector Length
-------------

.. doctest::

    >>> import pyvelox.pyvelox as pv
    >>> flat_vec = pv.from_list([1, 2, 3])
    >>> len(flat_vec)
    3
    >>> const_vec =  pv.constant_vector(10, 3)
    >>> len(const_vec)
    3
    >>> base_indices = [0, 0, 1, 0, 2]
    >>> dict_vec = pv.dictionary_vector(pv.from_list([1, 2, 3]), base_indices)
    >>> len(dict_vec)
    5
