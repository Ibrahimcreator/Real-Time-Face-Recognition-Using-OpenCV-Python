import cv2
import numpy as np
from PIL import Image
import os

class FaceRecognizerTrainer:
    def __init__(self, dataset_path):
        self.recognizer = cv2.face_LBPHFaceRecognizer.create()
        self.dataset_path = dataset_path

    def get_image_ids(self):
        image_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path)]
        faces = []
        ids = []

        for image_path in image_paths:
            face_image = Image.open(image_path).convert('L')
            face_np = np.array(face_image)
            user_id = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(face_np)
            ids.append(user_id)
            cv2.imshow("Training", face_np)
            cv2.waitKey(1)

        return ids, faces

    def train_recognizer(self):
        IDs, facedata = self.get_image_ids()
        self.recognizer.train(facedata, np.array(IDs))
        self.recognizer.write("TrainedModel.yml")
        cv2.destroyAllWindows()
        print("Training Completed............")

if __name__ == "__main__":
    dataset_path = "datasets"
    trainer = FaceRecognizerTrainer(dataset_path)
    trainer.train_recognizer()
