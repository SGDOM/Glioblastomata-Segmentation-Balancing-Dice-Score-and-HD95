# Glioblastomata-Segmentation--Balancing-Dice-Score-and-HD95


## Required environment
Install the requirements.txt dependencies

```
pip install -r requirements.txt
```

## Model Training and prediction
Dataset preparation

#### Datasets

BraTS 2021:http://braintumorsegmentation.org/. 

BraTS 2020: https://www.med.upenn.edu/cbica/brats2020/data.html



#### Dataset preprocessing and Load required functions

Running the Preprocessing.py

```
python Preprocessing.py
```

#### Model training and prediction

Running the Training & prediction.py
```
python Training & prediction.py
```
hyperparameters like learning rate and Focal Tverskey loss parameters varies over 50 epochs for swift convergence.

#### Simulation Results
Simulation findings and complexity evaluation are disclosed subsequent to the paper's publication.
