# Glioblastomata-Segmentation--Balancing-Dice-Score-and-HD95


## Required environment
Install the requirements.txt dependencies

```
pip install -r requirements.txt
```

## Model Training and Predection
Dataset preparation

#### Download dataset

https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1


#### Dataset preprocessing and Required functions

Running the Preprocessing.py

```python
python Preprocessing.py
```

#### Model training and Predection

Running the Training & predection.py
```python
python Training & predection.py       (hyper parameter like learning rate and Focal Tverskey loss parameters apla and beta varies over 50 epoch for swift convergence)
```

#### Simulation results
