import cv2

class FaceDatasetCollector:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def collect_dataset(self):
        user_id = input("Enter Number: ")
        count = 0

        while True:
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                file_name = f'datasets/User.{user_id}.{count}.jpg'
                cv2.imwrite(file_name, gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1)

            if count > 1000:
                break

    def release_capture(self):
        self.video.release()
        cv2.destroyAllWindows()
        print("Data Collection Completed")

if __name__ == "__main__":
    dataset_collector = FaceDatasetCollector()
    dataset_collector.collect_dataset()
    dataset_collector.release_capture()
