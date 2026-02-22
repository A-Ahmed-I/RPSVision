import torch
import torch.nn as nn


class RPSClassifier(nn.Module):
	"""
	A Convolutional Neural Network for classifying Rock-Paper-Scissors images.

	Architecture assumes an input size of (3, 244, 244).
	"""

	def __init__(self, in_channels: int, base_filters: int, num_classes: int):
		"""
		Args:
			in_channels (int): Number of input channels (e.g., 3 for RGB).
			base_filters (int): Number of filters in the first convolution layer.
			num_classes (int): Number of output classes.
		"""
		super().__init__()

		self.features = nn.Sequential(
				nn.Conv2d(in_channels, base_filters, kernel_size=3),
				nn.SiLU(),
				nn.MaxPool2d(2),

				nn.Conv2d(base_filters, base_filters * 2, kernel_size=3),
				nn.SiLU(),
				nn.MaxPool2d(2),

				nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3),
				nn.SiLU(),
				nn.MaxPool2d(2)
		)

		flattened_size = (base_filters * 4) * 28 * 28

		self.classifier = nn.Sequential(
				nn.Flatten(),
				nn.Linear(flattened_size, base_filters),
				nn.SiLU(),
				nn.Linear(base_filters, num_classes)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.classifier(x)
		return x