import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms
import os
normalizeG = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

normalizeOld = transforms.Normalize(
    # CIFAR10训练集的通道均值（R, G, B），用于归一化
    mean=(0.491399689874, 0.482158419622, 0.446530924224),
    # CIFAR10训练集的通道标准差（R, G, B）
    std=(0.247032237587, 0.243485133253, 0.261587846975))


def getdataloader(args, normalize=normalizeG):
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        ds.CIFAR10(
            root='/home/user/disk/code/Rs20CIFAR10/cifar-10-python/',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomCrop(32, 4),
                normalize
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    # 创建测试集数据加载器（参数说明同上）
    test_loader = torch.utils.data.DataLoader(
        ds.CIFAR10(
            '/home/user/disk/code/Rs20CIFAR10/cifar-10-python/',
            train=False,  # 加载测试集
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    return train_loader, test_loader





normalizeG_cifar100 =transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

root_path = os.path.expanduser('/home/user/disk/code/ResNet18CIFAR100/')

def getdataloader_cifar100(args, normalize=normalizeG_cifar100):
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        ds.CIFAR100(
            root=root_path,
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        ds.CIFAR100(
            root_path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    return train_loader, test_loader



from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class TinyImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # 获取原始数据
        item = self.hf_dataset[idx]

        image = item['image']
        label = item['label']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_tiny_imagenet_dataloaders(data_path='/home/user/disk_nvme/tiny-imagenet', batch_size=32, num_workers=4):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomCrop(64, 4),
        normalize])


    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])


    train_dataset = load_dataset(data_path, split='train')
    val_dataset = load_dataset(data_path, split='validation')


    train_dataset_wrapped = TinyImageNetDataset(train_dataset, transform=train_transform)
    val_dataset_wrapped = TinyImageNetDataset(val_dataset, transform=val_transform)


    train_loader = DataLoader(
        train_dataset_wrapped,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset_wrapped,
        batch_size=batch_size,
        shuffle=False,  # 验证集通常不需要shuffle
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader






def get_ImageNet_getdataloader(args, normalize=normalizeG):
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        ds.ImageNet(
            root='/home/user/disk_nvme/ImageNet',
            split='train',
            transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,

        **kwargs
    )

    # 创建测试集数据加载器（参数说明同上）
    test_loader = torch.utils.data.DataLoader(
        ds.ImageNet(
            root='/home/user/disk_nvme/ImageNet',
            split='val',
            transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,

        **kwargs
    )

    return train_loader, test_loader