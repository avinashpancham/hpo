MLflow Sandbox
==============

Train a ML model for sentiment classification while keeping track of the performance of the different models via MLflow
The ML model are built in Scikit-learn and will via a GridSearch evaluate different model architectures:
- Different vectorizer settings: ngram size
- Different classifiers and classifier settings: Naive Bayes, Random Forest and Support Vector Machines.

Subsequently the best model is selected from all the model based on performance.
The best model is then trained on the complete corpus.


============
Dependencies
============
This project has the following dependencies:

- Dependencies
    - `Git LFS <https://git-lfs.github.com/>_`
    - Python >= 3.7
    - `Poetry <https://python-poetry.org/>`
    - `Docker <https://www.docker.com/>`

=====
Setup
=====

1. Clone the repo

2. Create a virtual environment with at least Python 3.7 via the tool of your choice (conda, venv, etc.)

3. Install the Python dependencies with Poetry

.. code-block:: bash

   $ poetry install

4. Create the folders `database` and `artifact` in the data folder

.. code-block:: bash

   $ cd data
   $ mkdir database
   $ mkdir artifacts

============
Train models
============

1. Run MLflow server via the code shown below. This will run the Makefile to startup the Postgres DB and the MLflow server.
The MLflow server is accessible via localhost:5000

.. code-block:: bash

   $ make mlflow-server

2. Train the different ML models using Scikit-learn.
After the run is finished an experiment is made containing the statistics for all the runs.

.. code-block:: bash

   $ python train_hp_optimizer.py

3. Train the best model on the complete dataset and evaluate performance on the test dataset

.. code-block:: bash

   $ python train_best_model.py

4. The best model is stored in the folder `trained_model` in the folder with the corresponding experiment name.
The model.pkl is your trained ML model that can be utilized to make predictions!

