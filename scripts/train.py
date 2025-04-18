import os
import json
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import List, Tuple
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split
from torch.utils.data import TensorDataset, DataLoader
from monai import transforms
from monai.utils import set_determinism
from loguru import logger


# === helper functions ===
def save_json(object, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(object, f, ensure_ascii=False, indent=4)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# === config class ===
class Config():
    def __init__(
        self,
        data_id,
        result_id,
        batch_size=2,
        num_no_image=0,
        image_only=True,
        epochs=250,
        lr=0.01,
        seed=216,
        dropout=0.0,
        depths=[32, 64, 128, 256, 256, 64],
        test_size=0.2,
    ):
        self.data_id = data_id
        self.result_id = result_id
        self.batch_size = batch_size
        self.num_no_image = num_no_image
        self.image_only = image_only
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.dropout = dropout
        self.depths = depths
        self.test_size = test_size

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4
        )

    @classmethod
    def fromJSON(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            json_str = json.load(f)
        data = json.loads(json_str)
        
        return cls(
            data_id=data['data_id'],
            result_id=data['result_id'],
            batch_size=data['batch_size'],
            num_no_image=data['num_no_image'],
            image_only=data['image_only'],
            epochs=data['epochs'],
            lr=data['lr'],
            seed=data['seed'],
            dropout=data['dropout'],
            depths=data['depths'],
            test_size=data['test_size'],
        )
    

# === dataset class ===
class HeartDataset(Dataset):
    def __init__(self, data_dir, image_only=True, transforms=None, stats=None, permute=True):
        super().__init__()
        self.data = pd.read_csv(data_dir).dropna()
        self.image_only = image_only
        self.permute = permute

        if self.image_only:
            self.data = self.data[["Study ID", "image", "Age"]]
        else:
            self.non_image_columns = [
                col for col in self.data.columns 
                if col not in ["Study ID", "image", "Age"]
            ]

            # Compute stats if not provided
            if stats is None:
                self.height_mean = self.data["height"].mean()
                self.height_std = self.data["height"].std()
                self.weight_mean = self.data["weight"].mean()
                self.weight_std = self.data["weight"].std()
                self.stats = {
                    "height_mean": self.height_mean,
                    "height_std": self.height_std,
                    "weight_mean": self.weight_mean,
                    "weight_std": self.weight_std
                }
            else:
                self.stats = stats
                self.height_mean = stats["height_mean"]
                self.height_std = stats["height_std"]
                self.weight_mean = stats["weight_mean"]
                self.weight_std = stats["weight_std"]

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dir = self.data.iloc[idx]["image"]
        age = torch.tensor([self.data.iloc[idx]["Age"]], dtype=torch.float32)
        non_image_data = torch.tensor([0])

        if self.image_only:
            img = self.transforms(img_dir) if self.transforms else img_dir
        else:
            img = self.transforms(img_dir) if self.transforms else img_dir
            non_image_data = self.data.iloc[idx][self.non_image_columns]

            # Standardize height and weight using training stats
            non_image_data["Height"] = (non_image_data["Height"] - self.height_mean) / self.height_std
            non_image_data["Weight"] = (non_image_data["Weight"] - self.weight_mean) / self.weight_std

            # Convert non-image data to tensor
            non_image_data = non_image_data.values.astype('float32')
            non_image_data = torch.tensor(non_image_data, dtype=torch.float32)
        if self.permute:
            img = torch.permute(img, (0, 3, 2, 1)) # Channel, Axial, Coronal, Sagittal
        return img, non_image_data, age
    

# === model class ===
class RegressionSFCNTorch(nn.Module):
    def __init__(self, *, in_ch: int=1,
        dropout: float=.0,
        include_top: bool=True,
        depths: List[int]=[32, 64, 128, 256, 256, 64],
        prediction_range: Tuple[float, float]=(3.0, 100.0),
        num_non_image: int=0,
    ):
        super(RegressionSFCNTorch, self).__init__()
        self.include_top = include_top
        self.prediction_range = prediction_range
        self.num_non_image = num_non_image
        self.block1 = nn.Sequential(OrderedDict([
            ('block1_conv', nn.Conv3d(in_ch, depths[0], kernel_size=(3, 3, 3), stride=1, padding='same')),
            ('block1_norm', nn.BatchNorm3d(num_features=depths[0], momentum=0.01, eps=0.001)),
            ('block1_relu', nn.ReLU()),
            ('block1_pool', nn.MaxPool3d(kernel_size=(2, 2, 2)))
        ]))
        self.block2 = nn.Sequential(OrderedDict([
            ('block2_conv', nn.Conv3d(depths[0], depths[1], kernel_size=(3, 3, 3), stride=1, padding='same')),
            ('block2_norm', nn.BatchNorm3d(num_features=depths[1], momentum=0.01, eps=0.001)),
            ('block2_relu', nn.ReLU()),
            ('block2_pool', nn.MaxPool3d(kernel_size=(2, 2, 2)))
        ]))
        self.block3 = nn.Sequential(OrderedDict([
            ('block3_conv', nn.Conv3d(depths[1], depths[2], kernel_size=(3, 3, 3), stride=1, padding='same')),
            ('block3_norm', nn.BatchNorm3d(num_features=depths[2], momentum=0.01, eps=0.001)),
            ('block3_relu', nn.ReLU()),
            ('block3_pool', nn.MaxPool3d(kernel_size=(2, 2, 2)))
        ]))
        self.block4 = nn.Sequential(OrderedDict([
            ('block4_conv', nn.Conv3d(depths[2], depths[3], kernel_size=(3, 3, 3), stride=1, padding='same')),
            ('block4_norm', nn.BatchNorm3d(num_features=depths[3], momentum=0.01, eps=0.001)),
            ('block4_relu', nn.ReLU()),
            ('block4_pool', nn.MaxPool3d(kernel_size=(1, 2, 2)))
        ]))
        self.block5 = nn.Sequential(OrderedDict([
            ('block5_conv', nn.Conv3d(depths[3], depths[4], kernel_size=(3, 3, 3), stride=1, padding='same')),
            ('block5_norm', nn.BatchNorm3d(num_features=depths[4], momentum=0.01, eps=0.001)),
            ('block5_relu', nn.ReLU()),
            ('block5_pool', nn.MaxPool3d(kernel_size=(1, 2, 2)))
        ]))
        self.top = nn.Sequential(OrderedDict([
            ('top_conv', nn.Conv3d(depths[4], depths[5], kernel_size=(1, 1, 1), stride=1, padding='same')),
            ('top_norm', nn.BatchNorm3d(num_features=depths[5], momentum=0.01, eps=0.001)),
            ('top_relu', nn.ReLU()),
            ('top_pool', nn.AvgPool3d(kernel_size=(6, 4, 4))),
        ]))
        self.dropout = nn.Sequential(OrderedDict([
            ('top_dropout', nn.Dropout(p=dropout))
        ]))
        self.prediction = nn.Sequential(OrderedDict([
            ('predictions', nn.Linear(in_features=depths[5] + self.num_non_image, out_features=1))
        ]))

    def forward(self, x, x_non_image=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        if self.include_top:
            x = self.top(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        if self.num_non_image > 0:
            x = torch.cat((x, x_non_image), dim=1)
        x = self.prediction(x)
        x = x.reshape(-1,1)
        return x


# === training functions and classes ===
def train_step(model, dataloader, loss_fn, optimizer, device):
    # Put model in train mode
    model.to(device)
    model.train()
    # Set up train loss
    train_loss = 0
    train_mae = 0
    # Loop through data loader to get data batch
    for batch, (X, N, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.float().to(device), y.float().to(device)
        if N.any():
            N = N.float().to(device)
        # 1. Forward pass
        if N.any():
            y_pred = model(X, N)
        else:
            y_pred = model(X)
        # 2. Calculate and accumulate loss
        loss = loss_fn(y, y_pred)
        train_loss += loss.item()
        batch_mae = mean_absolute_error(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        train_mae += batch_mae
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
        logger.info(f"Batch number {(batch+1)}/{len(dataloader)}: Train MAE: {batch_mae} -- Train MSE: {loss.item()}")
    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_mae /= len(dataloader)
    return train_loss, train_mae


def test_step(model, dataloader, loss_fn, device):
    # Put model in eval model
    model.to(device)
    model.eval()
    # Setup test loss
    test_loss = 0
    test_mae = 0
    # Turn on inference context manager
    with torch.no_grad():
        for batch, (X, N, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.float().to(device), y.float().to(device)
            if N.any():
                N = N.float().to(device)
            # 1. Forward
            if N.any():
                test_pred = model(X, N)
            else:
                test_pred = model(X)
            # 2. Calculate and accumulate loss
            loss = loss_fn(y, test_pred)
            test_loss += loss.item()
            test_mae += mean_absolute_error(y.cpu().numpy(), test_pred.cpu().numpy())
        # Adjust metrics to get average loss per batch
        test_loss = test_loss / len(dataloader)
        test_mae = test_mae / len(dataloader)
    return test_loss, test_mae


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, save_dir, current_valid_loss, current_valid_mae, epoch, model, optimizer, loss_fn):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            logger.info(f"\nBest validation loss: {self.best_valid_loss:.5} | MAE: {current_valid_mae:.3f}")
            logger.info(f"\nSaving best model for epoch: {epoch + 1}\n")
            res = {'epoch': epoch + 1,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'loss': loss_fn}
            save_name = f'ep{epoch}_val-loss={current_valid_loss:.3f}_val-mae={current_valid_mae:.3f}.pth'
            os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
            torch.save(res, os.path.join(save_dir, "checkpoints", save_name))


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device, save_dir):
    save_best_model = SaveBestModel()

    # Create a dictionary to save the training progress
    results = {'train_loss': [], 'test_loss': [], "train_mae": [], "test_mae": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_mae = train_step(model=model, dataloader=train_dataloader,
                                loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_mae = test_step(model=model, dataloader=test_dataloader,
                              loss_fn=loss_fn, device=device)
        # Save the mest model till now if we have the least loss in the current epoch
        save_best_model(save_dir=save_dir, current_valid_loss=test_loss,
                        current_valid_mae=test_mae, epoch=epoch,
                        model=model, optimizer=optimizer, loss_fn=loss_fn)
        # Print out what's happening
        logger.info(f"Epoch: {epoch + 1} | train_loss: {train_loss:.5f} | test_loss: {test_loss:.5f} | train_mae: {train_mae:.3f} | test_mae: {test_mae:.3f}")
        # Update results dictionary
        results['train_loss'].append(train_loss)
        results['train_mae'].append(train_mae)
        results['test_loss'].append(test_loss)
        results['test_mae'].append(test_mae)

        # Save train_loss and test_loss results
        history_filename = os.path.join(save_dir, 'loss_results.csv')
        pd.DataFrame(results).to_csv(history_filename, index=False)
        logger.info(f'\nTrain loss and test loss history were saved in {history_filename}')


# === main function === 
if __name__ == "__main__":
    config = Config(
        data_id="processed_dummy_1",
        result_id="run_dummy_1",
        batch_size=8,
        num_no_image=0,
        image_only=True,
        epochs=4,
        lr=0.0005,
        seed=216,
        dropout=0.5,
        depths=[32, 64, 128, 256, 128, 64],
        test_size=0.2,
    )

    dataset_id = config.data_id

    results_id = config.result_id
    results_dir = f"../results/{results_id}/"
    os.makedirs(results_dir, exist_ok=True)

    num_non_image = config.num_no_image
    image_only = config.image_only
    epochs = config.epochs
    bs = config.batch_size
    lr = config.lr
    seed = config.seed
    depths = config.depths
    dropout = config.dropout
    test_size = config.test_size

    config_json = config.toJSON()
    save_json(config_json, os.path.join(results_dir, "config.json"))

    set_determinism(seed)

    img_transforms = transforms.Compose([
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        transforms.EnsureType(),
        transforms.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1, clip=True),
    ])

    # split dataset into train and valid
    df = pd.read_csv(f"../data/{dataset_id}/info/normal_heart.csv")
    train_df, valid_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_data_path = os.path.join(results_dir, "train_data.csv")
    valid_data_path = os.path.join(results_dir, "valid_data.csv")
    train_df.to_csv(train_data_path, index=False)
    valid_df.to_csv(valid_data_path, index=False)

    # Create datasets
    train_ds = HeartDataset(data_dir=train_data_path, image_only=image_only, transforms=img_transforms)
    valid_ds = HeartDataset(data_dir=valid_data_path, image_only=image_only, transforms=img_transforms)

    logger.info(f"Train size: {train_ds.__len__()}\tValid size: {valid_ds.__len__()}")

    # Create DataLoaders
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, drop_last=True)

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RegressionSFCNTorch(
        depths=depths,
        dropout=dropout,
        num_non_image=num_non_image,
    ).to(device)
    logger.info(f"Number of model parameters: {count_parameters(model)}")

    # Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train(model, train_dl, valid_dl, optimizer, loss_fn, epochs, device, results_dir)