import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np


def GetFER2013_for_search(args, new_batch_size=None):
    path_train = './data/FER_2013/train/'
    path_vaild = './data/FER_2013/PublicTest/'
    path_test = './data/FER_2013/PrivateTest/'

    batch_size = args.batch_size
    if new_batch_size:
        batch_size = new_batch_size

    transforms_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])
    transforms_vaild = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.ImageFolder(root=path_train, transform=transforms_train)
    vaild_data = torchvision.datasets.ImageFolder(root=path_vaild, transform=transforms_vaild)
    test_data = torchvision.datasets.ImageFolder(root=path_test, transform=transforms_vaild)

    num_all = len(train_data + vaild_data + test_data)
    print("FER 2013:", num_all)
    indices = list(range(num_all))
    split = int(np.floor(args.train_portion * num_all))

    train_queue = torch.utils.data.DataLoader(train_data + vaild_data + test_data, batch_size=batch_size,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),num_workers=2)
    valid_queue = torch.utils.data.DataLoader(train_data + vaild_data + test_data, batch_size=batch_size,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_all]), num_workers=2)

    return train_queue, valid_queue


def GetFER2013_for_retrain(args):
    path_train = './data/FER_2013/train/'
    path_vaild = './data/FER_2013/PublicTest/'
    path_test = './data/FER_2013/PrivateTest/'

    transforms_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])
    transforms_vaild = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.ImageFolder(root=path_train, transform=transforms_train)
    vaild_data = torchvision.datasets.ImageFolder(root=path_vaild, transform=transforms_vaild)
    test_data = torchvision.datasets.ImageFolder(root=path_test, transform=transforms_vaild)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(vaild_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,num_workers=2)

    return train_queue, valid_queue, test_queue


def GetFER2013_for_search_LBP(args, new_batch_size=None):
    path_train = './data/FER_2013/train_LBP/'
    path_vaild = './data/FER_2013/PublicTest_LBP/'
    path_test = './data/FER_2013/PrivateTest_LBP/'

    batch_size = args.batch_size
    if new_batch_size:
        batch_size = new_batch_size

    transforms_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])
    transforms_vaild = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.ImageFolder(root=path_train, transform=transforms_train)
    vaild_data = torchvision.datasets.ImageFolder(root=path_vaild, transform=transforms_vaild)
    test_data = torchvision.datasets.ImageFolder(root=path_test, transform=transforms_vaild)

    num_all = len(train_data + vaild_data + test_data)
    print("FER 2013:", num_all)
    indices = list(range(num_all))
    split = int(np.floor(args.train_portion * num_all))

    train_queue = torch.utils.data.DataLoader(train_data + vaild_data + test_data, batch_size=batch_size,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),num_workers=2)
    valid_queue = torch.utils.data.DataLoader(train_data + vaild_data + test_data, batch_size=batch_size,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_all]), num_workers=2)

    return train_queue, valid_queue


def GetFER2013_for_retrain_LBP(args):
    path_train = './data/FER_2013/train_LBP/'
    path_vaild = './data/FER_2013/PublicTest_LBP/'
    path_test = './data/FER_2013/PrivateTest_LBP/'

    transforms_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])
    transforms_vaild = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.ImageFolder(root=path_train, transform=transforms_train)
    vaild_data = torchvision.datasets.ImageFolder(root=path_vaild, transform=transforms_vaild)
    test_data = torchvision.datasets.ImageFolder(root=path_test, transform=transforms_vaild)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(vaild_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,num_workers=2)

    return train_queue, valid_queue, test_queue
