import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, ConfusionMatrixDisplay

class ModelTrainer:
	"""
	Handles the training, validation, evaluation, and export of a PyTorch model.
	"""

	def __init__(
			self,
			model: nn.Module,
			optimizer: torch.optim.Optimizer,
			criterion: nn.Module,
			train_loader: DataLoader,
			val_loader: DataLoader,
			epochs: int,
			checkpoint_path: str,
			device: Optional[str] = None
	):
		"""
		Args:
			model (nn.Module): The neural network model.
			optimizer (torch.optim.Optimizer): The optimizer (e.g., AdamW).
			criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
			train_loader (DataLoader): DataLoader for training data.
			val_loader (DataLoader): DataLoader for validation data.
			epochs (int): Number of training epochs.
			checkpoint_path (str): Path to save the best model weights.
			device (str, optional): Computation device ('cuda' or 'cpu'). Auto-detects if None.
		"""
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.epochs = epochs
		self.checkpoint_path = checkpoint_path

		self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)
		self.best_val_f1 = float("-inf")

	def _run_epoch(self, dataloader: DataLoader, is_training: bool) -> Tuple[float, float, float, List[int], List[int]]:
		"""
		Runs a single pass (training or evaluation) over a dataloader.

		Returns:
			Tuple: (average_loss, accuracy, f1_score, all_preds, all_labels)
		"""
		if is_training:
			self.model.train()
		else:
			self.model.eval()

		total_loss = 0
		all_preds = []
		all_labels = []

		context = torch.enable_grad() if is_training else torch.no_grad()

		with context:
			for images, labels in tqdm(dataloader, desc="Training" if is_training else "Eval", leave=False):
				images, labels = images.to(self.device), labels.to(self.device)

				if is_training:
					self.optimizer.zero_grad()

				outputs = self.model(images)
				loss = self.criterion(outputs, labels)

				if is_training:
					loss.backward()
					self.optimizer.step()

				total_loss += loss.item()
				preds = outputs.argmax(dim=1)

				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(labels.cpu().numpy())

		avg_loss = total_loss / len(dataloader)

		acc = np.mean(np.array(all_preds) == np.array(all_labels))
		f1 = f1_score(all_labels, all_preds, average="macro")

		return avg_loss, acc, f1, all_preds, all_labels

	def train_loop(self) -> Dict[str, List[float]]:
		"""
		Executes the full training loop over specified epochs.

		Returns:
			Dict: History of loss and metrics.
		"""
		history = {
			"Train Loss": [],
			"Validation Loss": [],
			"Train Accuracy": [],
			"Validation Accuracy": [],
			"Train F1": [],
			"Validation F1": []
		}

		LINE = 60

		print(f"Starting training on {self.device}...")

		for epoch in range(self.epochs):
			train_loss, train_acc, train_f1, _, _ = self._run_epoch(self.train_loader, is_training=True)
			val_loss, val_acc, val_f1, _, _ = self._run_epoch(self.val_loader, is_training=False)

			history["Train Loss"].append(train_loss)
			history["Validation Loss"].append(val_loss)
			history["Train Accuracy"].append(train_acc)
			history["Validation Accuracy"].append(val_acc)
			history["Train F1"].append(train_f1)
			history["Validation F1"].append(val_f1)

			print(f"\nEpoch [{epoch + 1}/{self.epochs}]")
			print("=" * LINE)
			print(f"{'Set':<10}{'Loss':<15}{'Acc (%)':<15}{'F1 (%)':<15}")
			print("-" * LINE)

			print(
					f"{'Train':<10}"
					f"{train_loss:<15.4f}"
					f"{train_acc * 100:<15.2f}"
					f"{train_f1 * 100:<15.2f}"
			)

			print(
					f"{'Val':<10}"
					f"{val_loss:<15.4f}"
					f"{val_acc * 100:<15.2f}"
					f"{val_f1 * 100:<15.2f}"
			)

			if val_f1 > self.best_val_f1:
				self.best_val_f1 = val_f1
				torch.save(self.model.state_dict(), self.checkpoint_path)
				print("-" * LINE)
				print(f"New Best Model Saved | Val F1: {val_f1 * 100:.2f}%")

			print("=" * LINE)

		return history

	def evaluate_test_set(self, test_loader: DataLoader, class_names: List[str]):
		"""
		Evaluates the model on a hold-out test set and plots a confusion matrix.
		"""
		try:
			self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
			print("Loaded best model weights for evaluation.")
		except FileNotFoundError:
			print("Checkpoint not found. Using current model weights.")

		loss, acc, f1, labels, preds = self._run_epoch(test_loader, is_training=False)

		print("\n" + "=" * 30)
		print(f"Test Accuracy: {acc:.2%}")
		print(f"Test F1 Score: {f1:.2%}")
		print("=" * 30)

		fig, ax = plt.subplots(figsize=(8, 8))
		ConfusionMatrixDisplay.from_predictions(
				labels, preds, display_labels=class_names, cmap="Blues", ax=ax
		)
		ax.set_title("Test Set Confusion Matrix")
		plt.show()

	def plot_history(self, history: Dict[str, List[float]]):
		"""Plots training curves."""
		epochs = range(1, len(history["Train Loss"]) + 1)

		plt.figure(figsize=(12, 5))

		# ===== Loss Plot =====
		plt.subplot(1, 3, 1)
		plt.plot(
				epochs, history["Train Loss"],
				marker="o", linestyle="-", linewidth=2,
				color="tab:blue", label="Train Loss"
		)
		plt.plot(
				epochs, history["Validation Loss"],
				marker="s", linestyle="--", linewidth=2,
				color="tab:red", label="Val Loss"
		)
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.title("Training vs Validation Loss")
		plt.legend()
		plt.grid(alpha=0.3)

		# ===== Accuracy Plot =====
		plt.subplot(1, 3, 2)
		plt.plot(
				epochs, history["Train Accuracy"],
				marker="o", linestyle="-", linewidth=2,
				color="tab:green", label="Train Accuracy"
		)
		plt.plot(
				epochs, history["Validation Accuracy"],
				marker="s", linestyle="--", linewidth=2,
				color="tab:orange", label="Val Accuracy"
		)
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy")
		plt.title("Training vs Validation Accuracy")
		plt.legend()
		plt.grid(alpha=0.3)

		# ===== F1 Plot =====
		plt.subplot(1, 3, 3)
		plt.plot(
				epochs, history["Train F1"],
				marker="o", linestyle="-", linewidth=2,
				color="tab:purple", label="Train F1"
		)
		plt.plot(
				epochs, history["Validation F1"],
				marker="s", linestyle="--", linewidth=2,
				color="tab:brown", label="Val F1"
		)
		plt.xlabel("Epochs")
		plt.ylabel("F1")
		plt.title("Training vs Validation F1")
		plt.legend()
		plt.grid(alpha=0.3)

		plt.tight_layout()
		plt.show()

	def export_onnx(self, output_path: str, input_shape: Tuple[int, int, int]):
		"""Exports the best model to ONNX format."""

		try:
			self.model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
		except FileNotFoundError:
			print("Warning: Checkpoint not found, exporting current model state.")

		self.model.eval().cpu()

		dummy_input = torch.randn(*input_shape)

		torch.onnx.export(
				self.model,
				dummy_input,
				output_path,
				opset_version=11,
				input_names=["input"],
				output_names=["output"],
				dynamic_axes={
					"input": {0: "batch"},
					"output": {0: "batch"}
				}
		)

		print(f"ONNX exported: {output_path}")