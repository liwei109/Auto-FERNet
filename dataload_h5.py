import numpy as np
import torch
from PIL import Image
import h5py
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


class FER2013(data.Dataset):
    """`FER2013 Dataset.
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('./data/FER_2013/data.h5', 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((28709, 48, 48))

        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_labels = self.data['PublicTest_label']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape((3589, 48, 48))

        else:
            self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_labels = self.data['PrivateTest_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3589, 48, 48))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)


def GetFER2013_for_search(args, new_batch_size=None):
    batch_size = args.batch_size
    if new_batch_size:
        batch_size = new_batch_size

    cut_size = 44
    print('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop(44),
        # transforms.Resize([32, 32]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(),
        # transforms.TenCrop(cut_size),
        # transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    trainset = FER2013(split='Training', transform=transform_train)
    PublicTestset = FER2013(split='PublicTest', transform=transform_train)
    PrivateTestset = FER2013(split='PrivateTest', transform=transform_train)

    num_all = len(trainset + PublicTestset + PrivateTestset)
    print("Num of FER 2013:", num_all)
    indices = list(range(num_all))
    split = int(np.floor(args.train_portion * num_all))
    train_queue = torch.utils.data.DataLoader(trainset + PublicTestset + PrivateTestset, batch_size=batch_size,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                              num_workers=1)
    valid_queue = torch.utils.data.DataLoader(trainset + PublicTestset + PrivateTestset, batch_size=batch_size,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                  indices[split:num_all]), num_workers=1)
    return train_queue, valid_queue


def GetFER2013_for_retrain(args):
    cut_size = 44
    print('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop(44),
        # transforms.Resize([32, 32]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(),
        # transforms.TenCrop(cut_size),
        # transforms.Resize([32, 32]),
        transforms.ToTensor(),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    trainset = FER2013(split='Training', transform=transform_train)
    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    PublicTestset = FER2013(split='PublicTest', transform=transform_test)
    valid_queue = torch.utils.data.DataLoader(PublicTestset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
    test_queue = torch.utils.data.DataLoader(PrivateTestset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_queue, valid_queue, test_queue


class CK(data.Dataset):
    """`CK+ Dataset.
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        there are 135,177,75,207,84,249,54 images in data
        we choose 123,159,66,186,75,225,48 images for training
        we choose 12,8,9,21,9,24,6 images for testing
        the split are in order according to the fold number
    """

    def __init__(self, split='Training', fold = 10, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.fold = fold # the k-fold cross validation
        self.data = h5py.File('./data/CK+48/CK_data.h5', 'r', driver='core')

        number = len(self.data['data_label']) #981
        sum_number = [0,135,312,387,594,678,927,981] # the sum of class number
        test_number = [12,18,9,21,9,24,6] # the number of each class

        test_index = []
        train_index = []

        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10: #the last fold start from the last element
                    test_index.append(sum_number[j]+(self.fold-1)*test_number[j]+k)
                else:
                    test_index.append(sum_number[j+1]-1-k)

        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        print(len(train_index),len(test_index))

        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = []
            self.train_labels = []
            for ind in range(len(train_index)):
                self.train_data.append(self.data['data_pixel'][train_index[ind]])
                self.train_labels.append(self.data['data_label'][train_index[ind]])

        elif self.split == 'Testing':
            self.test_data = []
            self.test_labels = []
            for ind in range(len(test_index)):
                self.test_data.append(self.data['data_pixel'][test_index[ind]])
                self.test_labels.append(self.data['data_label'][test_index[ind]])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Testing':
            img, target = self.test_data[index], self.test_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Testing':
            return len(self.test_data)


def GetCK(args, fold):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    trainset = CK(split='Training', fold=fold, transform=transform_train)
    testset = CK(split='Testing', fold=fold, transform=transform_test)

    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    test_queue = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=1)

    return train_queue, test_queue

def GetJAFFE(args):
    path_test = './data/JAFFE/test/'

    transforms_train = transforms.Compose([
        transforms.Grayscale(),  # 使用ImageFolder默认扩展为三通道，重新变回去就行
        transforms.RandomHorizontalFlip(),  # 随机翻转
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])
    transforms_vaild = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    test_data = torchvision.datasets.ImageFolder(root=path_test, transform=transforms_vaild)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,num_workers=2)

    return test_queue