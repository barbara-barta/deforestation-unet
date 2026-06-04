import numpy as np
import torch
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .utils import local_run


class DeforestationDataset(Dataset):
    
    def __init__(self, root_directory,image_name,label_name,indices = None, transform=None):
        self.root_directory = root_directory
        self.image_directory = self.root_directory / str(image_name)
        self.label_directory = self.root_directory / str(label_name)
        self.image_paths = sorted(self.image_directory.glob("*.tif"))
        self.label_paths = sorted(self.label_directory.glob("*.tif"))
        self.image_dict = {img.stem: img.name for img in self.image_paths}
        self.label_dict = {lbl.stem: lbl.name for lbl in self.label_paths}

        for key,value in self.image_dict.items():
            if key not in self.label_dict.keys():
                print(f"Warning: image {value} does not have a corresponding label.")

        self.transform = transform
        self.indices = indices

    def __len__(self):
        if self.indices:
            dataset_size = len(self.indices)
        else:
            dataset_size = len(self.image_paths)
        return dataset_size

    def __getitem__(self, index):
        if self.indices:
            img_path = self.image_paths[self.indices[index]]
            img_name = img_path.name
            lbl_path = self.label_directory / img_name
        else:
            img_path = self.image_paths[index]
            img_name = img_path.name
            lbl_path = self.label_directory / img_name
        
        with rio.open(img_path) as img:
            img = img.read().astype('float32')
            if self.transform:
                img = self.transform(img)

            lbl = rio.open(lbl_path)
            lbl = lbl.read().astype('float32')

            sample = [img,torch.from_numpy(lbl)]
            return sample


class min_max_scale():
    def __init__(self):
        pass

    def __call__(self,img):
        img_min,img_max = np.min(img),np.max(img)
        img_scaled =  (img - img_min)/(img_max-img_min + 1e-8)
        return img_scaled

class to_tensor():
    def __init__(self):
        pass
    def __call__(self,img):
        return torch.from_numpy(img)

train_transform = transforms.Compose([
    min_max_scale(),
    to_tensor()
])


def make_am4_datasets(base_dir):
    train_dataset_am4 = DeforestationDataset(base_dir / "data" / "AM4" / "raw" / "amazon" / "Training","image","label",transform = train_transform)
    test_dataset_am4 = DeforestationDataset(base_dir / "data" / "AM4" / "raw" / "amazon" / "Test","image","mask",transform = train_transform)
    val_dataset_am4 = DeforestationDataset(base_dir / "data" / "AM4" / "raw" / "amazon" / "Validation","images","masks",transform = train_transform)
    return train_dataset_am4, test_dataset_am4, val_dataset_am4


def make_am4_dataloaders(base_dir, batch_size=None, num_workers=None):
    if batch_size is None:
        if local_run():
            batch_size = 4
        else:
            batch_size = 32
    if num_workers is None:
        if local_run():
            num_workers = 0
        else:
            num_workers = 4

    train_dataset_am4, test_dataset_am4, val_dataset_am4 = make_am4_datasets(base_dir)

    sample = train_dataset_am4[0][0]
    #first sample, image only
    print(f"After transform, min of sample is {torch.min(sample)}, max is {torch.max(sample)}.") # check
    print(f"After transform, shape of sample is {sample.shape}") # check
    print(f"Size of train dataset is {len(train_dataset_am4)}, size of val dataset is {len(val_dataset_am4)}, size of test dataset is {len(test_dataset_am4)}.")

    train_dataloader_am4 = DataLoader(train_dataset_am4,batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)
    test_dataloader_am4 = DataLoader(test_dataset_am4,batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=True)
    val_dataloader_am4 = DataLoader(val_dataset_am4,batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=True)
    return train_dataset_am4, test_dataset_am4, val_dataset_am4, train_dataloader_am4, test_dataloader_am4, val_dataloader_am4
