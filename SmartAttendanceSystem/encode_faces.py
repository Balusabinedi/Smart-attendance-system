import face_recognition
import os
import cv2
import pickle

# Paths
dataset_path = "dataset"
encoding_file = "encodings.pickle"
known_encodings = []
known_names = []

# Loop over each person
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    # Loop over images
    for filename in os.listdir(person_folder):
        image_path = os.path.join(person_folder, filename)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save encodings to file
data = {"encodings": known_encodings, "names": known_names}
with open(encoding_file, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encodings generated and saved successfully.")
