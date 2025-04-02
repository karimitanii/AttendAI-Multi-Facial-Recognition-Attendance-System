import os
import face_recognition
import cv2
import sys
import pandas as pd
from flask import Flask, Response
from utils import (
    load_student_data,
    initialize_attendance,
    create_database_encodings,
    recognize_face,
    mark_attendance,
)
import dlib
import time

# Set working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_dir)

# Define the PID file path
pid_file_path = os.path.join(working_dir, "python_process.pid")

# Save the PID to a file
pid = os.getpid()
with open(pid_file_path, "w") as pid_file:
    pid_file.write(str(pid))

# Check if dlib is using CUDA (GPU)
print("Using CUDA for face recognition:", dlib.DLIB_USE_CUDA)

if not dlib.DLIB_USE_CUDA:
    print("Warning: CUDA is not being used. Make sure GPU support is properly configured.")

# Flask app to serve video feed
app = Flask(__name__)

# Global variables
video_capture = None
encodings = None
attendance_csv = None
student_data = None  # Fix: Declare globally


def setup(attendance_csv_path):
    global encodings, attendance_csv, student_data

    attendance_csv = attendance_csv_path  # ✅ No need to join again
    students_csv = os.path.join(working_dir, "students.csv")

    if not os.path.exists(students_csv):
        print(f"Error: Student data file {students_csv} not found.")
        sys.exit(1)

    student_data = load_student_data(students_csv)
    initialize_attendance(attendance_csv)

    print(f" Loaded student data from: {students_csv}")
    print(student_data.head())  # Debugging output

    encodings = create_database_encodings(student_data)


@app.route('/video_feed')
def video_feed():
    """ Serve the live camera feed via Flask. """
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def generate_frames():
    """ Generate frames from the live video feed and perform face recognition. """
    global video_capture, encodings, attendance_csv, student_data

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        recognized_name = recognize_face(encodings, frame)

        if recognized_name:
            mark_attendance(recognized_name, attendance_csv, student_data)

            face_locations = face_recognition.face_locations(frame)
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    recognized_name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def main():
    """ Main entry point to run the Flask app and handle attendance recording. """
    global video_capture

    if len(sys.argv) < 2:
        print("Usage: python main.py <course_id>")
        sys.exit(1)

    course_id = sys.argv[1]
    setup(course_id)

    time.sleep(2)
    video_capture = cv2.VideoCapture(0)

    print("Starting Flask server for live video feed...")
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)

    video_capture.release()
    cv2.destroyAllWindows()

    if os.path.exists(pid_file_path):
        os.remove(pid_file_path)
        print(" PID file removed. Model stopped successfully.")


if __name__ == "__main__":
    main()
