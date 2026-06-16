import cv2
import numpy as np
import tensorflow as tf

class FaceMaskPredictor:
    def __init__(self, model_path="models/best_model.h5"):
        self.loaded = False
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.loaded = True
            print(f"[FaceMaskPredictor] Model loaded from {model_path}")
        except Exception as e:
            print(f"[FaceMaskPredictor] WARNING: Could not load model from {model_path}: {e}")
            
        self.img_size = (224, 224)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def preprocess_image(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def _classify(self, preprocessed_img):
        """
        Returns (label, confidence).
        Model uses sigmoid output:
          - output close to 0 → with_mask   (class index 0, alphabetically first)
          - output close to 1 → without_mask (class index 1, alphabetically second)
        Keras image_dataset_from_directory sorts folders alphabetically:
          index 0 = with_mask, index 1 = without_mask
        """
        raw = float(self.model.predict(preprocessed_img, verbose=0)[0][0])
        if raw > 0.5:
            label = "Without Mask"
            confidence = raw                # probability of being without_mask
        else:
            label = "With Mask"
            confidence = 1.0 - raw          # probability of being with_mask
        return label, confidence

    def predict(self, image):
        if not self.loaded:
            return image, [{"label": "Model Not Found", "confidence": 0.0}]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        detections = []

        if len(faces) == 0:
            # No face detected — run on full image as fallback
            preprocessed = self.preprocess_image(image)
            label, confidence = self._classify(preprocessed)
            color = (0, 0, 255) if label == "Without Mask" else (0, 255, 0)
            cv2.putText(image, f"{label} ({confidence*100:.1f}%)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            detections.append({"label": label, "confidence": confidence})
            return image, detections

        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            preprocessed = self.preprocess_image(face_img)
            label, confidence = self._classify(preprocessed)
            color = (0, 0, 255) if label == "Without Mask" else (0, 255, 0)

            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{label}: {confidence*100:.1f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            detections.append({"label": label, "confidence": confidence})

        return image, detections
