Hyperparameter optimization
===========================

The right hyperparameters can make all the difference for the performance of your ML model,
but finding the optimal set of hyperparameters is often a slow process.
AutoML solutions have been made to assist in the search for the optimal set of hyperparameters, see for example `Cloud AutoML <https://cloud.google.com/automl?hl=nl>`_
This gives us an optimal model, but we do not know what the architecure and the corresponding optimal set of hyperparameter are.

Hyperparameter optimization (hpo) can luckily also be done using Python. In this repository it is demonstrated how to perform hpo in the:

- Naive way using gridsearch with Scikit-Learn's GridSearchCV
- Smart way using bayesian optimization with Optuna

As use case a binary sentiment classification task (positive vs negative) is chosen.
The ML model architecture is built via Scikit Learn and consist of a Scikit Learn pipeline with an Anonymizer (custom transformer),
TFIDF vectorizer and a classifier.
The performance of each model is tracked using MLflow. The hyperparameter searchspace in this situation has two dimensions:

- Different vectorizer settings: ngram ranges
- Different classifiers and classifier settings: Naive Bayes, Random Forest and Support Vector Machines.

After evaluating all the models their performance, the best model is selected. This model then trained on the complete corpus.


============
Dependencies
============
This project has the following dependencies:

- Dependencies
    - `Large Movie Review Dataset <http://ai.stanford.edu/~amaas/data/sentiment/>`_ from Stanford (already included in the repo)
    - `Git LFS <https://git-lfs.github.com/>`_
    - Python >= 3.7
    - `Docker <https://www.docker.com/>`_
    - Optional: `Poetry <https://python-poetry.org/>`_

=====
Setup
=====

1. Download Git LFS `here <https://git-lfs.github.com/>`_ or via brew:

.. code-block:: bash

   $ brew install git-lfs

2. Install Git LFS on your machine:

.. code-block:: bash

   $ sudo git lfs install --system --skip-repo

3. Clone the repo. If you have already cloned the repo before installing Git LFS, run the following to get all the *large files* (else only the pointers to the large files will be present on your machine):

.. code-block:: bash

   $ git lfs pull origin

4. Create a virtual environment with at least Python 3.7 via the tool of your choice (conda, venv, etc.)

5. Install the Python dependencies

Using poetry:

.. code-block:: bash

   $ poetry install

Not using poetry:

.. code-block:: bash

   $ pip install -r requirements.txt


5. Create the directories :code:`database` and :code:`artifacts` in the :code:`data` directory

.. code-block:: bash

   $ cd data
   $ mkdir database
   $ mkdir artifacts

============================
Explore hyperparameter space
============================

1. Start a MLflow server via the code shown below. This :code:`Makefile` command starts up the Postgres database and the MLflow UI.
The MLflow server is accessible at *localhost:5000*.

.. code-block:: bash

   $ make mlflow-server

With the current configuration the statistics are stored in the Postgres database, whereas the artifacts are stored on your disk.

2. Define the to be explore hyperparameter space. The default hyperparameters to be searched are:

- Vectorizer:
    - TFIDF vectorizer
        - ngram_range: (1, 1), (1, 2)
- Classifier:
    - SVM
        - C: [0.1, 0.2]
    - Multinomial Naive Bayes
        - alpha: [1e-2, 1e-1]
    - RandomForestClassifier
        - max depth: [2, 4]

3. Explore the hyperparameter space using either gridsearch or bayesian optimization

Using gridsearch:

.. code-block:: bash

   $ python hpo_gridsearch.py

The following arguments can be provided:

- --size or -s: sample size of the dataset; default: 25000
- --workers or -w: the number of CPU cores that can be used; default: 2
- --random or -r: if provided a randomsearch instead of a gridsearch will be performed. The hyperparameter space is randomly sampled for these combinations;  default: not specified

Using bayesian optimization:

.. code-block:: bash

   $ python hpo_bayesian.py

The following arguments can be provided:

- --size or -n: sample size of the dataset; default: 25000
- --workers or -w: the number of CPU cores that can be used; default: 2
- --trial or -t: the number of hyperparameters sets to explore; default: 20


4. After the run is finished the parameters and metrics (performance) of each model is
visible in the corresponding experiment in the MLFlow dashboard


5. Train the best model on the complete dataset and evaluate performance on the test dataset

.. code-block:: bash

   $ python train.py

6. The best model is stored in the directory :code:`trained_model` in the subdirectory with the corresponding experiment name.
The :code:`model.pkl` is your trained ML model that can be utilized to make predictions!

