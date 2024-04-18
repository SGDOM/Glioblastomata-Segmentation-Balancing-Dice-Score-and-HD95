# Glioblastomata-Segmentation--Balancing-Dice-Score-and-HD95


## Required environment
Install the requirements.txt dependencies

```
pip install -r requirements.txt
```

## Model Training and prediction
Dataset preparation

#### Download dataset

https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1


#### Dataset preprocessing and Required functions

Running the Preprocessing.py

```
python Preprocessing.py
```

#### Model training and prediction

Running the Training & prediction.py
```
python Training & prediction.py
```
hyperparameters like learning rate and Focal Tverskey loss parameters apla and beta varies over 50 epochs for swift convergence.

#### Simulation Results
Simulation findings and complexity evaluation are disclosed subsequent to the paper's publication.
