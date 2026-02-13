import torch
import polars as pl
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from src.data.custom_data import RPSDataset
from sklearn.model_selection import train_test_split


class DataLoaderFactory:
	"""
	Manages data splitting and the creation of PyTorch DataLoaders.
	"""

	def __init__(self, metadata: pl.DataFrame, class_map: Dict[str, int]):
		"""
		Args:
			metadata (pl.DataFrame): The source dataframe containing all data.
			class_map (Dict[str, int]): Mapping from label names to integers.
		"""
		self.metadata = metadata
		self.class_map = class_map

	def split_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
		"""
		Splits data into Train (70%), Validation (20%), and Test (10%).

		Returns:
			Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: (train_df, test_df, val_df)
		"""
		train_data, temp_data = train_test_split(self.metadata, test_size=0.3, random_state=0, shuffle=True,
		                                         stratify=self.metadata["Label"])

		test_data, val_data = train_test_split(temp_data, test_size=1 / 3, random_state=0, shuffle=True,
		                                       stratify=temp_data["Label"])

		return train_data, test_data, val_data

	def create_dataloaders(
			self,
			batch_size: int,
			target_size: Tuple[int, int]
	) -> Tuple[DataLoader, DataLoader, DataLoader]:
		"""
		Creates DataLoaders for train, test, and validation sets.

		Args:
			batch_size (int): Samples per batch.
			target_size (Tuple[int, int]): Image size for the dataset.

		Returns:
			Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, test_loader, val_loader)
		"""
		train_df, test_df, val_df = self.split_data()

		train_ds = RPSDataset(train_df, self.class_map, target_size)
		test_ds = RPSDataset(test_df, self.class_map, target_size)
		val_ds = RPSDataset(val_df, self.class_map, target_size)

		train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
		test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

		return train_loader, test_loader, val_loader