import face_recognition
import cv2
from utils import (
    load_student_data,
    initialize_attendance,
    create_database_encodings,
    recognize_face,
    mark_attendance,
)
import dlib

# Check if dlib is using CUDA (GPU)
print("Using CUDA for face recognition:", dlib.DLIB_USE_CUDA)

if not dlib.DLIB_USE_CUDA:
    print("Warning: CUDA is not being used. Make sure GPU support is properly configured.")


def main():
    # File paths
    students_csv = "students.csv"
    attendance_csv = "attendance.csv"

    # Load student data and initialize attendance
    student_data = load_student_data(students_csv)
    initialize_attendance(attendance_csv)

    print(f"Loaded student data from: {students_csv}")
    print(student_data.head())

    # Create face encodings for known students
    encodings = create_database_encodings(student_data)

    # Start live video feed for face recognition
    video_capture = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect and recognize faces in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            recognized_name = None
            for name, known_encoding in encodings.items():
                match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
                if match[0]:
                    recognized_name = name
                    mark_attendance(name, attendance_csv)
                    break

            # Draw a green box and display the name if a face is recognized
            top, right, bottom, left = face_location
            color = (0, 255, 0)  # Green
            color_2 = (0,0,255) #red
            thickness = 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

            if recognized_name:
                cv2.putText(
                    frame,
                    recognized_name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color_2,
                    thickness
                )

        # Display the camera feed
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()