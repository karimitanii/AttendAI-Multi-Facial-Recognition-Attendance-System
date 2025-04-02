from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Initialize the OpenCV video capture
video_capture = cv2.VideoCapture(0)

def generate_frames():
    """Generate frames from the webcam for the video feed."""
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of an MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the homepage."""
    return "<h1>Live Camera Feed</h1><p>Go to <a href='/video_feed'>/video_feed</a> to see the camera feed.</p>"

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Streams the video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)
