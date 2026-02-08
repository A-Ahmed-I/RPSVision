from pathlib import Path

# ======================================================
# Dataset & Paths Configuration
# ======================================================
base_path = Path("/media/ahmed/Data/RPSVision/data")

checkpoint_path = "/media/ahmed/Data/RPSVision/checkpoint/best_model.pth"
output_path = "/media/ahmed/Data/RPSVision/checkpoint/best_model.onnx"


# ======================================================
# Classes & Label Mapping
# ======================================================
class_name = ["paper", "rock", "scissors"]

class_map = {
    "paper": 0,
    "rock": 1,
    "scissors": 2,
}


# ======================================================
# Dataset Schema & Visualization Settings
# ======================================================
schema = ["File Path", "Label"]

num_samples = 6
cols = 3
figsize = (12, 8)

path_col = schema[0]
label_col = schema[1]


# ======================================================
# Image Preprocessing
# ======================================================
target_size = (244, 244)


# ======================================================
# Training Hyperparameters
# ======================================================
batch_size = 32
epochs = 30

lr = 1e-4
weight_decay = 1e-5


# ======================================================
# Model Input & Architecture Configuration
# ======================================================
input_shape = (1, 3, 244, 244)

in_channels = 3
base_filters = 16

# ======================================================
# ONNX Runtime Settings
# ======================================================
onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]