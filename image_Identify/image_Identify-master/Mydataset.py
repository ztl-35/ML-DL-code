import torch.utils.data as data
class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)