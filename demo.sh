# MNIST
python mnist.py

# HDGM
python hdgm.py --exptype power

# HIGGS
python higgs.py

# Blob with KDE
python blob_kde.py --exptype power
# Blob with GMM
python blob_gmm.py --exptype power

# Case Study: learning with noisy labels
python CIFAR10.py --noise_type 'symmetric'
python CIFAR10.py --noise_type 'pairflip'