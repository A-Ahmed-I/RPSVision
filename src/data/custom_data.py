import torch
import polars as pl
from typing import Dict, Tuple
from torch.utils.data import Dataset
from src.utils.helper import load_image



class RPSDataset(Dataset):
	"""
	A PyTorch Dataset for loading Rock-Paper-Scissors images from a Polars DataFrame.
	"""

	def __init__(
			self,
			data: pl.DataFrame,
			class_map: Dict[str, int],
			target_size: Tuple[int, int]
	):
		"""
		Args:
			data (pl.DataFrame): DataFrame containing 'file_path' and 'label' columns.
			class_map (Dict[str, int]): Dictionary mapping string labels to integer targets.
			target_size (Tuple[int, int]): Dimensions to resize images to (width, height).
		"""
		self.data = data
		self.class_map = class_map
		self.target_size = target_size

	def __len__(self) -> int:
		return self.data.height

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		row = self.data.row(index, named=True)

		path = row["File Path"]
		label = row["Label"]

		img_array = load_image(path, self.target_size)
		label_idx = self.class_map[label]

		img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)
		label_tensor = torch.tensor(label_idx, dtype=torch.long)

		return img_tensor, label_tensor