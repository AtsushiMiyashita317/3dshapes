import torch


class Dataset3DShapes(torch.utils.data.Dataset):
    def __init__(self, images=None, labels=None, indices=None):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = self.images[self.indices[idx]]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 255.0

        if self.labels is None:
            return img

        label = self.labels[self.indices[idx]]
        label = torch.from_numpy(label).float()
        color = label[0:3].mul(2 * torch.pi)
        color = torch.stack([color.sin(), color.cos()], dim=-1)  # (3, 2)
        scale = label[3:4]
        shape = label[4].long()
        orientation = label[5].div(30)
        return img, color, scale, shape, orientation

