import torchvision.transforms as transforms
import torchvision
import torch

def CK48_dataload(args, batch_size):
    path_test = './data/CK+48/test/'

    transforms_vaild = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    data_test = torchvision.datasets.ImageFolder(root=path_test, transform=transforms_vaild)

    test_queue = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)

    return test_queue

