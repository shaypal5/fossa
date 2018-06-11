fossa |fossa_icon|
##################
|PyPI-Status| |PyPI-Versions| |Build-Status| |Codecov| |LICENCE|

.. |fossa_icon| image:: https://github.com/shaypal5/fossa/blob/be1f8e84d311f926fd39e8ea421525782b4cb39f/fossa.png 

Distribution-based anomaly detection for time series.

.. code-block:: python

  >>> from fossa import LastStepPredictor
  ...

.. contents::

.. section-numbering::


Installation
============

.. code-block:: bash

  pip install fossa
  


Features
========

* Adheres to the ``scikit-learn`` classifier API.
* Pickle-able classifier objects.
* Pure python.
* Supports Python 3.5+.
* Fully tested.


Use
===

TBA


Contributing
============

Package author and current maintainer is Shay Palachy (shay.palachy@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed.

Installing for development
----------------------------

Clone:

.. code-block:: bash

  git clone git@github.com:shaypal5/fossa.git


Install in development mode, including test dependencies:

.. code-block:: bash

  cd fossa
  pip install -e '.[test]'



Running the tests
-----------------

To run the tests use:

.. code-block:: bash

  cd fossa
  pytest


Adding documentation
--------------------

The project is documented using the `numpy docstring conventions`_, which were chosen as they are perhaps the most widely-spread conventions that are both supported by common tools such as Sphinx and result in human-readable docstrings. When documenting code you add to this project, follow `these conventions`_.

.. _`numpy docstring conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`these conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

Additionally, if you update this ``README.rst`` file,  use ``python setup.py checkdocs`` to validate it compiles.


Credits
=======

Created by Shay Palachy (shay.palachy@gmail.com).


.. |PyPI-Status| image:: https://img.shields.io/pypi/v/fossa.svg
  :target: https://pypi.python.org/pypi/fossa

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/fossa.svg
   :target: https://pypi.python.org/pypi/fossa

.. |Build-Status| image:: https://travis-ci.org/shaypal5/fossa.svg?branch=master
  :target: https://travis-ci.org/shaypal5/fossa

.. |LICENCE| image:: https://github.com/shaypal5/fossa/blob/master/mit_license_badge.svg
  :target: https://github.com/shaypal5/fossa/blob/master/LICENSE
  
.. https://img.shields.io/github/license/shaypal5/fossa.svg

.. |Codecov| image:: https://codecov.io/github/shaypal5/fossa/coverage.svg?branch=master
   :target: https://codecov.io/github/shaypal5/fossa?branch=master
