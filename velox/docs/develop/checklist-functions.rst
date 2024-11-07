=================================================
Check the items when you add or modify a function
=================================================

Pull request title
------------------

* The title should be ``Add Spark[Presto] function xxx``.

Pull requeste description
-------------------------
The PR description will be your commit message, please pay attention to it.

* Add the link of Spark implementation for Spark function.
* Add the description of this function.
* Document the change you make, especially you move a function from Presto to the common lib, document the difference with Presto.
* Summarize the performance improvement if have.
* Attach the relevent issue like ``Resolves #1182link``.

Document
--------

* Document the added function following alphabetical order.
* Add the SQL example for each case.
* Use ``x`` format for the variable.
* Use different variable name for the input variables and the return type.
* Describe the null input behavior.

Comment
-------

* Add the comment for each public function. Including brief description and @param.
* For multiple line comments on headers for classes, exposed as documentation, use ///.
* Start with a capital letter and ending with a period (.).

Registration
------------

* Register similar functions in adjacent locations.
* Prefer the simple function even vector function has better performance gain.

Code behavior
-------------

Common code behavior
~~~~~~~~~~~~~~~~~~~~
* Use ``const auto`` if possible.
* Use ``auto*`` for the pointer.
* Function
    * Mark the function of class as const if possible.
    * Place the function, class, etc. to .cpp file if possible.
    * Function in anonymous namespace is static, do not add the ``static`` keyword.
    * Function of class is inline, do not add the ``inline`` keyword.
    * Check the function scope, ``private``, ``protected`` or ``public``.
* Argument and member
    * Use ``kEnableSpill`` format for the constant.
    * Place the field member of class after the functions.
    * Use ``_input`` format as struct constructor's input argument and ``input`` format as member.
    * Use ``input`` format as class constructor's input argument and ``input_`` format as member.
    * Use ``const arg_type<int32_t>&`` as function input rather than its actual type such as ``StringWriter``.
    * Use ``/*unused*/`` for unused input argument and name it properly.
    * Place ``const`` member before other members.
* ``explicit`` keyword is used for constructors that take a ``single`` argument.
* Add destructor for class if it is a parent class.
* Don't use anonymous namespace in header file, use the ``namespace detail {`` instead, but don't use ``detail::func`` in other where.
* Revise the included header files if you do some refactors.
* Use ``vector_size_t`` for vector row number and ``column_index_t`` for column index.
* Remove ``else if`` or ``else`` branch if returning early.
* Check the empty line between functions.
* Add the empty line to the end of new file.

Unit test code behavior
~~~~~~~~~~~~~~~~~~~~~~~

* Add the test if you move a function to public scope.
* Cover all the code branch, pay attention to the exception test, use ``VELOX_ASSERT_THROW`` rather than ``ASSERT_THROW`` in most cases.
* Use the ``const auto xx = lambda function`` to test, rather than add a function to class, unless the function will be reused.
* Test body format: ``TEST_F(ColumnWriterTest, testDecimalWriter)``
