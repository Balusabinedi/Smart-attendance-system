import cv2
import face_recognition
import pickle
import os
import pandas as pd
from datetime import datetime

# Load encodings
if os.path.exists("encodings.pickle"):
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)
else:
    print("[ERROR] No encodings file found. Please register at least one student.")
    exit()

# Mark attendance function
def mark_attendance(name):
    attendance_file = "attendance.csv"

    # Check if file exists and has content
    if not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(attendance_file, index=False)

    df = pd.read_csv(attendance_file)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Avoid duplicate entry for the same day
    if not ((df["Name"] == name) & (df["Date"] == date_str)).any():
        new_row = {"Name": name, "Date": date_str, "Time": time_str}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"[INFO] Attendance marked for {name}")
    else:
        print(f"[INFO] Attendance already marked for {name} today")

# Start recognition
print("[INFO] Starting camera. Press 'q' to exit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdx = [i for (i, b) in enumerate(matches) if b]
            name = data["names"][matchedIdx[0]]
            mark_attendance(name)
            break

    # Show camera feed
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
