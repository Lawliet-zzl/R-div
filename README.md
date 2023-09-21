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
To run HDGM experiment, choose --vtype = vjs, --n = 100,1000,1500,2500, and --d = 3,5,10,15,20 and run
```python
python hdgm.py --exptype power
```

## Case Study: learning with noisy labels
To run CIFAR10 experiment with symmetry flipping labels
```python
python CIFAR10.py --noise_type 'symmetric'
```
To run CIFAR10 experiment with pair flipping labels
```python
python CIFAR10.py --noise_type 'pairflip'
```
