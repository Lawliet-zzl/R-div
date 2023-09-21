# Code for "R-divergence for Estimating Model-oriented Distribution Discrepancy"

## HIGGS
Get the HIGGS data from here - https://drive.google.com/file/d/1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc/view

To run HIGGS experiment, choose --n = 500,1000,1500,2500,4000,5000 and run

```python
python higgs.py
```

## MNIST
Get the fake MNIST data from here - https://drive.google.com/file/d/13JpGbp7PEm4PfZ6VeqpFiy0lHfVpy5Z5/view

To run MNIST experiment, choose --n = 100,200,300,400,500 and run

```python
python mnist.py
```

## Blob
To run Blob experiment with KDE
```python
python blob_kde.py --exptype power
```
To run Blob experiment with GMM
```python
python blob_gmm.py --exptype power
```

## HDGM

## Case Study: learning with noisy labels
