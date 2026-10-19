import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from pathlib import Path
import json
import os
seed = 0

class SSDataset(Dataset):
    def __init__(self, dir, dataset, file_list = None, multiplier=1.0, downsample_rate=1, task="sleep_staging", return_file_name=False):
        self.dir = dir
        self.dataset = dataset
        self.multiplier = multiplier
        self.downsample_rate = downsample_rate
        self.task = task
        self.return_file_name = return_file_name
        if file_list:
            # If file list is provided, use only those files
            self.files = [self.dir / file_name for file_name in file_list]
        else:
            # If no file list is provided, follow the previous file selection logic
            if "shhs" in dataset and "mesa" in dataset:
                exclude_path = "single_stage_files.json"
                exclude_list = json.load(open(exclude_path)) if os.path.exists(exclude_path) else []
                print(len(exclude_list))
                # 2. 在列表推导式中加入过滤条件
                self.files = [file for file in self.dir.glob("*.npz")]
                print(len(self.files), '#############')
            elif "shhs_ecg" in dataset:
                self.files = [
                    file
                    for file in self.dir.glob("shhs-*")
                ]
            elif "shhs" in dataset:
                self.files = [
                    file
                    for file in self.dir.glob("shhs1*")
                    if file.name != "shhs1-204822.npz"
                ]
            elif "dreamt" in dataset:
                self.files = [file for file in self.dir.glob("*.npz")]
            elif "mesa" in dataset:
                self.files = [file for file in self.dir.glob("*.npz")]
                print("MESA contains files: ", len(self.files))
            

    def __len__(self):
        
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)
        ibi = data["data"].flatten() * self.multiplier # Remove the second dimension
        labels = data["stages"]
        fs = data["fs"].item()
        try:
            ahi = data["ahi"].item()
        except:
            ahi = 0

        # If necessary downsample to 25Hz
        # but extract dataset is in 25Hz already, so need not downsample in normal cases
        ibi = ibi[::self.downsample_rate]
        labels = labels[::self.downsample_rate]

        # Calculate the number of samples for each sequence element and number of ibi segments
        num_samples = int((fs / self.downsample_rate) * 30) # for 30 second segments
        num_segments = len(ibi) // num_samples

        # Reshape the ibi and labels array
        ibi = ibi[:num_samples*num_segments]
        ibi = ibi.reshape(num_segments, num_samples)
        labels = labels[::num_samples]
        labels = labels[:num_segments]
        labels = self.remap_labels(labels, self.task)

        if num_segments > 1100:
            ibi = ibi[:1100]
            labels = labels[:1100]
            num_segments = 1100

        # Convert data to PyTorch tensors
        ibi = torch.from_numpy(ibi).float()
        labels = torch.from_numpy(labels).long()

        if self.return_file_name:
            return ibi, labels, num_segments, ahi, file_path.stem
        else:
            return ibi, labels, num_segments, ahi

    def remap_labels(self, labels, task):
        # 0=Wake, 1=N1, 1=N2, 2=N3, 3=REM, 5=Movement => mapped to -1, plus -1 => -1 for out-of-range
        if task == "sleep_staging":
            label_map = {0:0, 1:1, 2:1, 3:3, 4:2, 5:-1, -1:-1}
        elif task == "sleep_wake":
            label_map = {0:0, 1:1, 2:1, 3:1, 4:1, 5:-1, -1:-1}

        # If a label is not in label_map keys, default to -1
        # e.g., label_map.get(x, -1) means "use label_map[x] if it exists, otherwise -1."
        remapped_labels = np.vectorize(lambda x: label_map.get(x, -1))(labels)
        return remapped_labels

    @staticmethod
    def collate_fn(batch):
        ibis, labels, lengths, ahis = zip(*batch)
        lengths = torch.tensor(lengths)

        ibis_padded = pad_sequence(ibis, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

        return ibis_padded, labels_padded, lengths, ahis


def create_dataloaders(dir, train_ratio, val_ratio, batch_size, num_workers, dataset, multiplier=1.0, downsampling_rate=1, task="sleep_staging"):
    """ Create train, validation, and test data loaders
    test_ratio = 1 - train_ratio - val_ratio

    Args:
        dir (_type_): _description_
        train_ratio (_type_): _description_
        val_ratio (_type_): _description_
        batch_size (_type_): _description_
        num_workers (_type_): _description_
        dataset (_type_): _description_
        multiplier (float, optional): _description_. Defaults to 1.0.
        downsampling_rate (int, optional): _description_. Defaults to 1.
        task (str, optional): _description_. Defaults to "sleep_staging".

    Returns:
        _type_: _description_
    """
    # Initialize the ibi dataset
    dataset = SSDataset(dir=dir, dataset=dataset, multiplier=multiplier, downsample_rate=downsampling_rate, task=task)
    # Create train/val/test split
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # Construct the data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn
    )

    return train_loader, val_loader, test_loader


def create_dataloaders_kfolds(
    dir,
    dataset,
    num_folds=5,
    val_ratio=0.1,
    batch_size=16,
    num_workers=8,
    multiplier=1.0,
    downsampling_rate=1,
):
    # Initialize the ibi dataset
    print(dir)
    print(dataset)
    dataset = SSDataset(
        dir=dir, multiplier=multiplier, dataset=dataset, downsample_rate=downsampling_rate
    )

    kfold = KFold(n_splits=num_folds, shuffle=True)
    folds = []

    for train_idx, test_idx in kfold.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)

        # Create train/val split
        train_size = int(len(train_dataset))
        val_size = int(val_ratio * len(dataset))
        train_size = train_size - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=SSDataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=SSDataset.collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=SSDataset.collate_fn,
        )
        folds.append((train_loader, val_loader, test_loader))

    return folds

# Test function
def test_dataloader_folds(dataloader_folds):
    for fold_index, (train_loader, val_loader, test_loader) in enumerate(dataloader_folds):
        print(f"Testing Fold {fold_index + 1}")

        # Test each loader
        for loader_name, loader in zip(['Train', 'Validation', 'Test'], [train_loader, val_loader, test_loader]):
            print(f"Loader: {loader_name}")
            for i, (X, y, lengths, AHIs) in enumerate(loader):
                print(f"Batch {i + 1}")
                print("X shape:", X.shape)  # Shape of the input batch
                print("y shape:", y.shape)  # Shape of the label batch
                print("Lengths:", lengths)  # Length of each sequence in the batch
                print("AHI scores: ", AHIs)
                if i == 0:  # Only print for the first batch to avoid excessive output
                    break
