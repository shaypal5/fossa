fossa |fossa_icon|
##################
|PyPI-Status| |PyPI-Versions| |Build-Status| |Codecov| |LICENCE|

.. |fossa_icon| image:: https://github.com/shaypal5/fossa/blob/88d480fd90820ea58c062029ce7e926201794e47/fossa_small.png

Distribution-based anomaly detection for time series data.

.. code-block:: python

  >>> from fossa import LatestWindowAnomalyDetector
  >>> clf = LatestWindowAnomalyDetector(p_threshold=0.005, power=0)
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


Approach
========

``fossaa`` is written to detect anomalies in multi-class (aka multi-categorical) frequency data in constant-sized time windows.

Power Divergence-based Anomaly Detection
----------------------------------------

The base approach of ``fossa`` to anomaly detection is non-parametric; when a model is fit with historic data split into constant-sized time windows, rather than generalizing it into some parametric model, the historic data - or a subset thereof - is kept in a simple data structure.

Then, when a fitted model is required to detect anomalies in some newer time window, it performs multiple Cressie-Read power divergence statistic and goodness of fit tests (using `the scipy implementation <http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.stats.power_divergence.html#scipy.stats.power_divergence>`_ between the frequency distribution in each (or some) of the historic windows and the frequency distribution in the new time window.

The test logic is as follows:

- For each of the historic windows in consideration, both its distribution and that of the new window are padded to the union of their categories; let us assume this results in `n` categories.
  - Then, for each of these `n` categories a one-vs-all binary distribution is generated for both the historic time window and the new one, and a power divergence test is performed between the two distributions; a rejection of the null hypothesis (occuring when the resulting p-value is smaller than some pre-defined :math: `\alpha`) means a trend is detected for that category (with the direction determined by the direction of the difference in the relative freqencies between the historic and new distribution), while otherwise no trend is detected.
  - Possibly, one additional power divergence test is performed between the original (non-binarized) categorical distributions of the historic and the new time window. The result of this test can be taken into account when determining the final conclusion of the comparison of this specific historic window and the new one, for any or all of the categories.
- If several historic time windows were used, their "votes" are somehow resolved into a final decision.


Use
===

Data Format
-----------

All anomaly detectors are desgined to receive as fit parameter a ``pandas`` DataFrame with a two-leveled multi-index, the first indexing time and the second indexing category/topic frequency per-window, and a single column of a numeric dtype, giving said frequency.

When detecting trends a similarly-indexed dataframe with detection results is returned, giving detected trends per time windows and category.


API
---

All anomaly detector objects in ``fossa`` have an identical API:

- ``fit`` - Recieves a history of time-windowed distributions to train on and fits the detector on it (see the `Data Format`_ section for the exact format). The set of categories may be different across different time windows or between historic and time windoes for detection; detection is done for the union of of categories over all commitee and new time windows.
- ``partial_fit`` - The same as ``fit``, but can also incrementaly fit an already-fit detector without necessarilly ignoring all past fitted data. Detectors who do not support incremental fitting will raise a ``NotImplementedError`` exception when this method is called.
- ``detect_trends`` - Recieves a new dataframe (in the correct format) and detects, for each of the time windows in it, trends for each category. In addition to the ``direction`` column - indicating trend direction, with -1 for a downward trend, 0 for no trend and 1 for an upward trend - the returned dataframe might contain additional columns detailing detection confidence or probability, like p-values or commitee vote results.
- ``predict`` - Like ``detect_trends``, except the returned dataframe always contains only a single column of detected trend directions.
   

Anomaly Detectors
-----------------

Committee-based Anomaly Detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This family of anomaly detectors all operate similarly: Every detector compares new time windows to a set of committe windows that represent its idea for relevant history and characteristic behaviour of the data; one detector might look at the same hour on the same weekday across several weeks, while another might look at all the same hours in the last 10 or 20 days, or the preciding few hours.

For each of the time windows given to the ``detect_trends`` or ``predict`` methods, a one-vs-all distribution is generated for each of the categories in the window (and is possibly normalized, depending on the specific detector and its initialization parameters). Then, for each of this distributions power divergence tests are performed between it and the corresponding distributions in each of the commitee time windows. Each commitee member "votes" on whether a trend is detected or not, and a decision is generated by some pre-set voting rule (for example, majority vote).


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
