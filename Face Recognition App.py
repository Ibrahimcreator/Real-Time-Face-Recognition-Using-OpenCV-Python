import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        self.root.geometry("1920x1080")

        self.setup_ui()

    def setup_ui(self):
        self.title_font = ("Arial", 24, "bold")
        self.label_font = ("Arial", 16)
        self.button_font = ("Arial", 18, "bold")
        self.text_color = "navy"

        self.project_label = tk.Label(self.root, text="Face Recognition Using OpenCV (LBPH Face Recognizer Algorithm)", font=self.title_font, fg=self.text_color)
        self.project_label.pack(pady=20)

        self.credits_label = tk.Label(self.root, text="Developed By: RamKishour M, K Prudhvi, Polapally Srikanth, Pobbathi Ganesh\nGuided by: Mrs. R.C. Dyana Priyadharshini", font=self.label_font, fg=self.text_color)
        self.credits_label.pack(pady=20)

        self.start_recognition_button = tk.Button(self.root, text="Start Face Recognition", command=self.start_recognition, font=self.button_font, bg="limegreen", fg="white")
        self.start_recognition_button.pack(pady=30)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit, font=self.button_font, bg="red", fg="white")
        self.exit_button.pack()

    def start_recognition(self):
        video = cv2.VideoCapture(0)

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        recognizer = cv2.face_LBPHFaceRecognizer.create()
        recognizer.read("TrainedModel.yml")

        name_list = ["", "RamKishour", "RamKishour", "RamKishour", "RamKishour", "Vijay", "Virat Kohli"]

        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                label, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if confidence > 50:
                    name = name_list[label]
                else:
                    name = "Unknown"

                # Draw rectangles and display names on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green text

            cv2.imshow("Face Recognition App", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
