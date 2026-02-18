import cv2
import math
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import Tuple

def load_image(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Loads an image from disk, converts it to RGB, resizes it,
    and normalizes pixel values to the range [0, 1].

    Args:
        path (str): The absolute or relative path to the image file.
        target_size (Tuple[int, int]): The desired (width, height) for resizing.

    Returns:
        np.ndarray: The processed image array with shape (H, W, 3) and values float32.

    Raises:
        FileNotFoundError: If the image cannot be read from the specified path.
    """

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0

    return img


def visualize_dataset_samples(
        metadata: pl.DataFrame,
        num_samples: int,
        cols: int,
        figsize: Tuple[int, int],
        path_col: str,
        label_col: str
) -> None:
    """
    Randomly samples images from the metadata DataFrame and plots them in a grid.

    Args:
        metadata (pl.DataFrame): DataFrame containing image paths and labels.
        num_samples (int, optional): Number of images to display. Defaults to 6.
        cols (int, optional): Number of columns in the grid. Defaults to 3.
        figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (12, 8).
        path_col (str, optional): Name of the column containing file paths. Defaults to "file_path".
        label_col (str, optional): Name of the column containing labels. Defaults to "label".
    """
    n = min(num_samples, len(metadata))
    samples = metadata.sample(n).to_dicts()

    rows = math.ceil(n / cols)

    plt.figure(figsize=figsize)

    for i, row in enumerate(samples):
        path = row.get(path_col)
        label = row.get(label_col)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label}", fontsize=12, fontweight="bold")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
