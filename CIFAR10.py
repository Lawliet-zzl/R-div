import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pickle

import os
import sys
import random
import argparse
import numpy as np
import csv
from PIL import Image
from cutils import download_url, check_integrity, noisify
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch PACS Training')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--ntrial', type=int, default=10, help="number of trials (default: 10)")

parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1, help=" (default: 0.1)")

parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--alpha', type = float, default = 0.4, help='(0,1), essential')
parser.add_argument('--beta', type = float, default = 0.3, help='[0, 0.5]')

parser.add_argument('--pre', default=False, action='store_true', help='load pretrained model')
args = parser.parse_args()

class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar10'
        self.noise_type=noise_type
        self.nb_classes=10

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            #if noise_type is not None:
            if noise_type !='clean':
                # noisify train data
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(train_labels=self.train_labels, 
                    noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                # _train_labels=[i[0] for i in self.train_labels]
                self.train_labels=[i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(self.train_labels)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type !='clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, self.train_noisy_labels[index], self.train_labels[index]
        else:
        	return img, self.test_labels[index], self.test_labels[index]
        # return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def setup_seed(seed = 3047):
    os.environ['PYTHONNASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def adjust_learning_rate(optimizer, epoch, decay):
	if epoch in decay:
		for param_group in optimizer.param_groups:
			new_lr = param_group['lr'] * 0.1
			param_group['lr'] = new_lr

def load_model(net, pth_name):
	net.load_state_dict(torch.load(pth_name))

def save_model(model, pth_name):
	torch.save(model.state_dict(), pth_name)

def pretrain(model, dataloader, optimizer, criterion):
    model.train()
    for epoch in tqdm(range(200)):
        for idx, (images, noisy, labels) in enumerate(dataloader):
            loss = criterion(model(images.cuda()), noisy.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        adjust_learning_rate(optimizer, epoch, [100, 150])

def test(model, criterion, dataloader):
	model.eval()
	test_loss = 0.0
	test_acc = 0.0
	total = 0
	with torch.no_grad():
		for idx, (images, noisy, labels) in enumerate(dataloader):
			images, noisy, labels = images.cuda(), noisy.cuda(), labels.cuda()
			targets = labels
			outputs = model(images)
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			test_acc += (predicted == targets).sum().item()
			test_loss += loss.item()*targets.size(0)
	test_loss = test_loss / total
	test_acc = 100.*test_acc/ total
	return test_loss, test_acc

def retrain(model_pretrained, model, dataloader, optimizer, criterion):
    model.train()
    model_pretrained.eval()
    noisy_acc, noisy_recall, vdiv = 0.0, 0.0, 0.0

    for epoch in tqdm(range(args.epoch)):
        train_loss = 0
        correct = 0
        total = 0
        for idx, (images, noisy, labels) in enumerate(dataloader):
            images, noisy, labels = images.cuda(), noisy.cuda(), labels.cuda()
            targets = noisy
            indexes_noisy, indexes_clean, noisy_acc_b, noisy_recall_b, vdiv_b = divide_batch(images, noisy, labels, model_pretrained(images), criterion)

            noisy_acc += noisy_acc_b
            noisy_recall += noisy_recall_b
            vdiv += vdiv_b

            outputs = model(images)
            loss_clean = criterion(outputs[indexes_clean,:], targets[indexes_clean])
            loss_noisy = criterion(outputs[indexes_noisy,:], targets[indexes_noisy])
            loss = args.beta * loss_noisy + (1 - args.beta) * loss_clean

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss/idx
        train_acc = 100.*correct/total
        adjust_learning_rate(optimizer, epoch, [100, 150])

    epoch, idx = epoch + 1, idx + 1
    noisy_acc = noisy_acc / (idx * epoch)
    noisy_recall = noisy_recall / (idx * epoch)
    vdiv = vdiv / (idx * epoch)

    return train_loss, train_acc, noisy_acc, noisy_recall, vdiv

def divide_batch(images, noisy, labels, outputs, criterion):
    m = nn.LogSoftmax(dim = 1)
    outputs = m(outputs)
    loss_values = -torch.gather(outputs, dim=1, index=noisy.unsqueeze(1)).squeeze(1)
    indexes = torch.argsort(loss_values, descending=True)
    num = int(outputs.size(0) * args.alpha)
    indexes_noisy = indexes[0:num]
    indexes_clean = indexes[num: outputs.size(0)]

    loss_clean = criterion(outputs[indexes_clean,:], noisy[indexes_clean])
    loss_noisy = criterion(outputs[indexes_noisy,:], noisy[indexes_noisy])
    vdiv = abs(loss_clean.item() - loss_noisy.item())

    TP = (noisy[indexes_noisy] != labels[indexes_noisy]).sum().item()
    TN = (noisy[indexes_clean] == labels[indexes_clean]).sum().item()

    noisy_acc = 100.*(TP + TN) / outputs.size(0)
    noisy_recall = 100.*TP / (noisy != labels).sum().item()

    return indexes_noisy, indexes_clean, noisy_acc, noisy_recall, vdiv

def get_loader():
    train_dataset = CIFAR10(root='./data', download=True,  
        train=True, transform=transforms.ToTensor(),
        noise_type=args.noise_type,
        noise_rate=args.noise_rate)

    test_dataset = CIFAR10(root='./data', download=True,  
        train=False, transform=transforms.ToTensor(),
        noise_type=args.noise_type,
        noise_rate=args.noise_rate)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=args.batch_size, drop_last=True,shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size=100, drop_last=True,shuffle=False)

    return train_loader, test_loader

def init_file(log_name):
    if not os.path.exists(log_name):
        with open(log_name, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            info = ['Noise Type', 'Noise Rate',
            'Seed', 'Epoch', 'lr', 'Alpha', 'Beta',
            'Pretrained Acc', 'Retrained Acc',
            'Noisy Acc', 'Noisy Recall', 'Divegence']
            logwriter.writerow(info)

def main():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("pth"):
        os.makedirs("pth")

    torch.manual_seed(args.seed)
    filename = 'CIFAR10_' + args.noise_type + '_' + str(args.noise_rate)
    log_name = ('logs/CIFAR10_' + args.noise_type + '.csv')
    init_file(log_name)

    model_pretrained = models.resnet18().cuda()
    model = models.resnet18().cuda()
    criterion =  nn.CrossEntropyLoss()

    optimizer_pretrained = optim.SGD(model_pretrained.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_retrained = optim.SGD(model.parameters(), lr=args.lr / (1 - args.alpha), momentum=0.9, weight_decay=5e-4)

    train_loader, test_loader = get_loader()

    print('Pretraining: ', filename)
    if args.pre:
        load_model(model_pretrained, './pth/CIFAR10.pth')
    else:
        pretrain(model_pretrained, train_loader, optimizer_pretrained, criterion)
        save_model(model_pretrained, './pth/CIFAR10.pth')
    test_loss_pre, test_acc_pre = test(model_pretrained, criterion, test_loader)

    print('Retraining: ', filename)
    train_loss, train_acc, noisy_acc, noisy_recall, vdiv = retrain(model_pretrained, model, train_loader, optimizer_retrained, criterion)
    save_model(model_pretrained, './pth/' + filename + '.pth')

    print('Testing ', filename)
    test_loss_re, test_acc_re = test(model, criterion, test_loader)

    info = [args.noise_type, args.noise_rate,
    args.seed, args.epoch, args.lr, args.alpha, args.beta,
    test_acc_pre, test_acc_re,
    noisy_acc, noisy_recall, vdiv]
    print(info)
    with open(log_name, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(info)

if __name__ == "__main__":
	main()