import cv2
import numpy as np
import onnxruntime as ort
from src.constant.constant import onnx_providers

class HandGestureRecognizer:
	def __init__(self, model_path, labels):
		self.labels = labels

		try:
			self.session = ort.InferenceSession(
					model_path, providers=onnx_providers
			)
		except Exception as e:
			print(f"Error loading model: {e}")
			exit()

		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self.session.get_outputs()[0].name
		self.input_shape = self.session.get_inputs()[0].shape
		self.target_size = (self.input_shape[2], self.input_shape[3])


	def preprocess(self, img_crop):
		"""
        Converts crop to ONNX input
        format:
        1. Resize to target size (e.g., 224x224)
        2. Normalize (0-1)
        3. Transpose HWC -> CHW (if model requires it)
        4. Add Batch Dimension
        """
		# Resize
		img = cv2.resize(img_crop, self.target_size)

		# Color space adjustment (BGR -> RGB)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Normalize to 0-1 (Standard for most models)
		img = img.astype(np.float32) / 255.0

		# Transpose to (Channels, Height, Width)
		# OpenCV uses HWC, PyTorch/ONNX usually expects CHW
		img = np.transpose(img, (2, 0, 1))

		# Add Batch Dimension: (1, 3, 224, 224)
		img = np.expand_dims(img, axis=0)

		return img


	def predict(self, hand_img):
		input_tensor = self.preprocess(hand_img)

		# Run Inference
		output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

		# Softmax
		probs = np.exp(output - np.max(output))
		probs = probs / probs.sum(axis=1, keepdims=True)

		pred_idx = np.argmax(probs, axis=1)[0]
		confidence = probs[0][pred_idx]

		return self.labels[pred_idx], confidence


