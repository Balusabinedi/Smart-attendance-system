import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import pickle
import pandas as pd
import cv2
import face_recognition
from datetime import datetime
from PIL import Image, ImageTk

# File paths
encodings_file = "encodings.pickle"
attendance_file = "attendance.csv"
students_file = "students.csv"

# Helper Functions
def load_encodings():
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_encodings(data):
    with open(encodings_file, "wb") as f:
        pickle.dump(data, f)

def load_students():
    if os.path.exists(students_file):
        return pd.read_csv(students_file)
    return pd.DataFrame(columns=["Name", "Registered On"])

def save_students(df):
    df.to_csv(students_file, index=False)

def mark_attendance(name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        if list(df.columns) != ["Name", "Time"]:
            df = pd.DataFrame(columns=["Name", "Time"])
    else:
        df = pd.DataFrame(columns=["Name", "Time"])

    df.loc[len(df)] = [name, timestamp]
    df.to_csv(attendance_file, index=False)

def register_student(name, image):
    data = load_encodings()
    students = load_students()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    if encodings:
        data["encodings"].append(encodings[0])
        data["names"].append(name)
        save_encodings(data)

        students.loc[len(students)] = [name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        save_students(students)
        messagebox.showinfo("Success", f"Student {name} registered!")
    else:
        messagebox.showerror("Error", "No face detected. Try again.")

def delete_student(name):
    data = load_encodings()
    students = load_students()

    indices_to_remove = [i for i, n in enumerate(data["names"]) if n == name]

    if not indices_to_remove:
        messagebox.showerror("Error", f"Student {name} not found.")
        return

    for i in sorted(indices_to_remove, reverse=True):
        del data["names"][i]
        del data["encodings"][i]

    save_encodings(data)
    students = students[students["Name"] != name]
    save_students(students)
    messagebox.showinfo("Success", f"Student {name} deleted successfully!")

def take_attendance():
    data = load_encodings()

    video = cv2.VideoCapture(0)
    known_encodings = data["encodings"]
    known_names = data["names"]

    while True:
        ret, frame = video.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    name = known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
                mark_attendance(name)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Attendance Taken Successfully!")

# UI Functions
def open_register():
    name = simpledialog.askstring("Register Student", "Enter student name:")
    if name:
        video = cv2.VideoCapture(0)
        ret, frame = video.read()
        video.release()
        if ret:
            register_student(name, frame)

def open_delete():
    name = simpledialog.askstring("Delete Student", "Enter student name to delete:")
    if name:
        delete_student(name)

def show_students():
    students = load_students()
    if students.empty:
        messagebox.showinfo("Students", "No students registered yet!")
    else:
        student_list = "\n".join([f"{row['Name']} (Registered: {row['Registered On']})" for index, row in students.iterrows()])
        messagebox.showinfo("Registered Students", student_list)

def show_attendance():
    if not os.path.exists(attendance_file):
        messagebox.showinfo("Attendance", "No attendance records yet.")
        return

    df = pd.read_csv(attendance_file)
    if df.empty:
        messagebox.showinfo("Attendance", "No attendance records yet.")
    else:
        records = "\n".join([f"{row['Name']} - {row['Time']}" for index, row in df.iterrows()])
        messagebox.showinfo("Attendance Records", records)

# UI Setup
root = tk.Tk()
root.title("Smart Attendance System - Premium Version")
root.geometry("700x500")
root.configure(bg="#f5f5f5")

header = tk.Label(root, text="Smart Attendance System", font=("Helvetica", 24, "bold"), fg="#333", bg="#f5f5f5")
header.pack(pady=20)

btn_frame = tk.Frame(root, bg="#f5f5f5")
btn_frame.pack(pady=20)

register_btn = tk.Button(btn_frame, text="Register New Student", font=("Helvetica", 14), command=open_register, width=25, bg="#4CAF50", fg="white")
register_btn.grid(row=0, column=0, padx=10, pady=10)

take_attendance_btn = tk.Button(btn_frame, text="Take Attendance", font=("Helvetica", 14), command=take_attendance, width=25, bg="#2196F3", fg="white")
take_attendance_btn.grid(row=0, column=1, padx=10, pady=10)

delete_btn = tk.Button(btn_frame, text="Delete Student", font=("Helvetica", 14), command=open_delete, width=25, bg="#f44336", fg="white")
delete_btn.grid(row=1, column=0, padx=10, pady=10)

attendance_btn = tk.Button(btn_frame, text="View Attendance Records", font=("Helvetica", 14), command=show_attendance, width=25, bg="#FFC107", fg="white")
attendance_btn.grid(row=1, column=1, padx=10, pady=10)

# Student List Button with Icon
students_img = Image.open("students.jpg")
students_img = students_img.resize((40, 40))
students_photo = ImageTk.PhotoImage(students_img)

students_btn = tk.Button(root, image=students_photo, command=show_students, bg="#f5f5f5", borderwidth=0)
students_btn.place(x=640, y=20)

root.mainloop()












