import cv2
import time
import mediapipe as mp
from src.constant.constant import *
from src.gestures.recognizer import HandGestureRecognizer


def main():
	# 1. Setup MediaPipe
	mp_hands = mp.solutions.hands
	hands = mp_hands.Hands(
			static_image_mode=False,
			max_num_hands=1,
			model_complexity=1,  # 0=Fast, 1=Balanced
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
	)
	mp_draw = mp.solutions.drawing_utils

	# 2. Setup Camera
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	# 3. Load Logic
	recognizer = HandGestureRecognizer(output_path, class_name)

	print("System Ready. Press 'q' to exit.")

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		h, w, _ = frame.shape
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Inference MediaPipe
		results = hands.process(rgb_frame)

		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:

				# Get Bounding Box Coordinates
				x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
				y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

				x_min, x_max = min(x_list), max(x_list)
				y_min, y_max = min(y_list), max(y_list)

				# --- Make Bounding Box Square ---
				box_w, box_h = x_max - x_min, y_max - y_min
				diff = abs(box_w - box_h) // 2

				if box_w > box_h:
					y_min -= diff
					y_max += diff
				else:
					x_min -= diff
					x_max += diff

				# Add Padding
				padding = 20
				x_min = max(0, x_min - padding)
				y_min = max(0, y_min - padding)
				x_max = min(w, x_max + padding)
				y_max = min(h, y_max + padding)

				# Crop Hand
				hand_crop = frame[y_min:y_max, x_min:x_max]

				# Validate Crop
				if hand_crop.size > 0 and hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
					# ONNX Inference
					label, conf = recognizer.predict(hand_crop)

					time.sleep(2)

					# Draw UI
					color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
					cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

					text = f"{label} {int(conf * 100)}%"

					# Text Background for readability
					(t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
					cv2.rectangle(frame, (x_min, y_min - 25), (x_min + t_w, y_min), color, -1)
					cv2.putText(frame, text, (x_min, y_min - 5),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

				# Draw Skeleton
				mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


		cv2.imshow("Gesture Recognition", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
