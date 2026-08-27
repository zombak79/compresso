Installation
============

Compresso is published as the ``compresso-pytorch`` distribution and imported
as ``compresso`` in Python code.

Install from PyPI
-------------------

.. code-block:: bash

   pip install compresso-pytorch

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/zombak79/compresso.git
   cd compresso
   pip install .

Development Install
-------------------

For local development, install the package in editable mode with the test extra:

.. code-block:: bash

   pip install -e ".[test]"

Build the Documentation Locally
-------------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   pip install -e .
   sphinx-build -b html docs/source docs/build/html

The generated HTML will be available in ``docs/build/html``.
