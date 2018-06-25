fossa |fossa_icon|
##################
|PyPI-Status| |PyPI-Versions| |Build-Status| |Codecov| |LICENCE|

.. |fossa_icon| image:: https://github.com/shaypal5/fossa/blob/88d480fd90820ea58c062029ce7e926201794e47/fossa_small.png

Distribution-based anomaly detection for time series.

.. code-block:: python

  >>> from fossa import LastWindowAnomalyDetector
  >>> clf = LastWindowAnomalyDetector(p_threshold=0.005, normalize=True)
  >>> clf.fit(historic_data_df)
  >>> clf.predict(new_data)
                       direction
  date       category
  2018-06-01 hockey          1.0
             footbal         0.0
             soccer         -1.0
             tennis          0.0


.. contents::

.. section-numbering::


Installation
============

.. code-block:: bash

  pip install fossa
  


Features
========

* ``scikit-learn``-like classifier API.
* Pickle-able classifier objects.
* Pure python.
* Supports Python 3.5+.
* Fully tested.


Use
===

TBA


Contributing
============

Current package maintainer (and one of the authors) is Shay Palachy (shay.palachy@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed.

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

Created by Shay Palachy (shay.palachy@gmail.com) and Omri Mendels.


.. |PyPI-Status| image:: https://img.shields.io/pypi/v/fossa.svg
  :target: https://pypi.org/project/fossa

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/fossa.svg
   :target: https://pypi.org/project/fossa

.. |Build-Status| image:: https://travis-ci.org/shaypal5/fossa.svg?branch=master
  :target: https://travis-ci.org/shaypal5/fossa

.. |LICENCE| image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://pypi.python.org/pypi/pdpipe

.. |Codecov| image:: https://codecov.io/github/shaypal5/fossa/coverage.svg?branch=master
   :target: https://codecov.io/github/shaypal5/fossa?branch=master
