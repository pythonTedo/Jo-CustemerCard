# Data Science - Project
## Data Vizualization, Exploration and Modeling

In this repository I am providing my work on Filialdaten. This data was provided by *Unser Ö-Bonus Club GmbH* as part of interview project.
In the **datainspect.ipynb** I am exploring the data.

### Clone the repo and create venv
```python
python -m venv ./jovenv
source ./jovenv/bin/activate
pip install -r requitements.txt
```
### Create a SQLite db file
In order to import data to SQL database you need to run load_data.py.
For this experiment, you will use SQLite verasion

```python
python load_data.py

```
### Run cluster_data.py
To run the pipline file cluster_data.py which will load the data, validate and preprocess. After that will be used in UMAP dimesionality reduction method (Uniform Manifold Approximation and Projection for Dimension Reduction) and then loaded with DBSCAN in order to create its own clusters of data based on points density.
DBSCAN uses two hyperparamethers - epsilon and min_samples which are being used to determine number of clusers which will be created and minimum points neighborhood each one in order to be a core point

```python
python cluster_data.py (epsilon_param) (min_samples)
```