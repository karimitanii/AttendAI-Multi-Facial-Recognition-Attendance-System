import pandas as pd
import face_recognition
from datetime import datetime
import os
import cv2

# ✅ Load student data from CSV (Fix: Ensure correct column names, no spaces, and case-insensitive matching)
def load_student_data(csv_path):
    df = pd.read_csv(csv_path, dtype={"student_id": str})  # ✅ Read IDs as strings
    df.columns = df.columns.str.strip()  # ✅ Remove spaces from column names
    df["name"] = df["name"].str.strip()  # ✅ Remove spaces from names
    return df

# ✅ Initialize attendance CSV (Ensure correct format)
def initialize_attendance(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠ Attendance file {csv_path} not found. Creating a new one.")
        df = pd.DataFrame(columns=["Student ID", "Student Name", "Date", "Status", "Timestamp"])
        df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"Student ID": str})  # ✅ Ensure IDs are read as strings

# ✅ Create encodings for known students
def create_database_encodings(student_df):
    print("Creating face encodings for students...")
    encodings = {}
    for _, row in student_df.iterrows():
        image_path = row["image_path"].strip()  # ✅ Trim spaces
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found. Skipping...")
            continue
        # ✅ Load and encode the student's face
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            encodings[row["name"]] = face_encodings[0]  # ✅ Store encoding by name
        else:
            print(f"Warning: No face found in image {image_path}. Skipping...")
    return encodings

# ✅ Recognize a face by comparing encodings
def recognize_face(encodings, input_frame):
    try:
        face_locations = face_recognition.face_locations(input_frame)
        face_encodings = face_recognition.face_encodings(input_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            for name, known_encoding in encodings.items():
                match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
                if match[0]:
                    return name.strip()  # ✅ Ensure no extra spaces
    except Exception as e:
        print(f"Error recognizing face: {e}")
    return None

# ✅ Fixed: Mark attendance correctly
def mark_attendance(student_name, attendance_path, student_data):
    try:
        # ✅ Ensure attendance file exists
        if not os.path.exists(attendance_path):
            print(f"⚠ Attendance file {attendance_path} not found. Creating a new one.")
            df = pd.DataFrame(columns=["Student ID", "Student Name", "Date", "Status", "Timestamp"])
            df.to_csv(attendance_path, index=False)

        # ✅ Read attendance file, ensuring consistent column format
        df = pd.read_csv(attendance_path, dtype={"Student ID": str})
        df.columns = df.columns.str.strip()  # ✅ Remove spaces from headers

        # ✅ Standardize student name format
        student_name = student_name.strip()

        # ✅ Get current date and timestamp
        current_date = datetime.now().strftime("%d-%m-%Y")
        current_timestamp = datetime.now().strftime("%H:%M:%S")

        # ✅ Lookup student ID using a case-insensitive match
        student_info = student_data[student_data["name"].str.lower() == student_name.lower()]

        if student_info.empty:
            print(f"❌ Error: No student ID found for {student_name}. Skipping...")
            return

        student_id = student_info["student_id"].astype(str).values[0]  # ✅ Extract single ID

        # ✅ **Prevent duplicate entries for the same student on the same date**
        is_present_today = ((df["Student ID"].astype(str) == student_id) & (df["Date"] == current_date)).any()
        if is_present_today:
            print(f"⚠ {student_name} ({student_id}) is already marked present today. Skipping duplicate entry.")
            return

        # ✅ **Ensure correct order of columns**
        new_entry = pd.DataFrame([{
            "Student ID": student_id,
            "Student Name": student_name,
            "Date": current_date,
            "Status": "Present",
            "Timestamp": current_timestamp
        }])

        # ✅ **Fix column mismatch issue**
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(attendance_path, index=False)

        print(f"✅ Marked {student_name} ({student_id}) as present.")

    except Exception as e:
        print(f"❌ Error marking attendance: {e}")
