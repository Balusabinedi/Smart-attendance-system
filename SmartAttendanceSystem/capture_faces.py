import cv2
import os

# Create a directory for saving face images
def create_folder(name):
    path = f"dataset/{name}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Capture face images
def capture_faces():
    name = input("Enter your name: ").strip()
    path = create_folder(name)

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(f"{path}/{name}_{count}.jpg", face)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, f"Image {count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Capturing Face Images", frame)
        if cv2.waitKey(1) == 27 or count >= 50:  # ESC key or 50 images
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved 50 images to {path}")

if __name__ == "__main__":
    capture_faces()
