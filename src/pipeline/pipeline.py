import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
from src.data.metadata import DatasetMetadataLoader
from src.data.dataloader import DataLoaderFactory
from src.model.classifier import RPSClassifier
from src.training.train import ModelTrainer


class TrainingPipeline:
	"""
	Orchestrates the full deep learning workflow:
	- Dataset metadata creation
	- DataLoader preparation
	- Model initialization
	- Training, validation, and testing
	- Model exporting (ONNX)
	"""

	def __init__(
			self,
			base_path: str,
			categories: List[str],
			class_map: Dict[str, int],
			checkpoint_path: str,
			epochs: int,
			batch_size: int,
			lr: float,
			weight_decay: float,
			in_channels: int,
			base_filters: int,
			target_size: Tuple[int, int],
			input_shape: Tuple[int, int, int, int],
			output_path: str,
	) -> None:
		"""
		Initializes the training pipeline configuration.

		Args:
			base_path (str): Root directory of the dataset.
			categories (List[str]): List of class folder names.
			class_map (Dict[str, int]): Mapping from class name to label index.
			checkpoint_path (str): Path to save the best model checkpoint.
			epochs (int): Number of training epochs.
			batch_size (int): Batch size for training and evaluation.
			lr (float): Learning rate.
			weight_decay (float): Weight decay for optimizer.
			in_channels (int): Number of input channels.
			base_filters (int): Number of base convolution filters.
			target_size (Tuple[int, int]): Target image size (H, W).
			input_shape (Tuple[int, int, int, int]):
				Model input shape in the form (B, C, H, W).
			output_path (str): Path to export the ONNX model.
		"""
		self.base_path = base_path
		self.categories = categories
		self.class_map = class_map
		self.checkpoint_path = checkpoint_path
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = lr
		self.weight_decay = weight_decay
		self.in_channels = in_channels
		self.base_filters = base_filters
		self.target_size = target_size
		self.input_shape = input_shape
		self.output_path = output_path

	def run(self) -> None:
		"""
		Executes the full training pipeline from data loading
		to evaluation and ONNX export.
		"""

		# 1. Build Metadata
		print("--- Building Metadata ---")
		# Assuming DatasetMetadataLoader was defined in previous steps
		meta_loader = DatasetMetadataLoader(self.base_path)
		metadata_df = meta_loader.to_dataframe(self.categories)
		print(f"Total samples found: {len(metadata_df)}")

		# 2. Create DataLoaders
		print("--- Creating DataLoaders ---")
		# Assuming DataLoaderFactory was defined in previous steps
		loader_factory = DataLoaderFactory(metadata_df, self.class_map)
		train_dl, test_dl, val_dl = loader_factory.create_dataloaders(batch_size=self.batch_size,
		                                                              target_size=self.target_size)

		# 3. Setup Model, Criterion, Optimizer
		print("--- Initializing Model ---")
		# Assuming RPSClassifier was defined in previous steps
		model = RPSClassifier(in_channels=self.in_channels, base_filters=self.base_filters,
		                      num_classes=len(self.categories))

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

		# 4. Train Model
		print("--- Starting Training ---")
		trainer = ModelTrainer(
				model=model,
				optimizer=optimizer,
				criterion=criterion,
				train_loader=train_dl,
				val_loader=val_dl,
				epochs=self.epochs,
				checkpoint_path=self.checkpoint_path
		)

		history = trainer.train_loop()

		# 5. Evaluate and Export
		print("--- Evaluating Results ---")
		trainer.plot_history(history)
		trainer.evaluate_test_set(test_dl, class_names=list(self.class_map.keys()))
		trainer.export_onnx(self.output_path, self.input_shape)