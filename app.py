from flask import Flask, render_template, Response
import cv2
from proctor import detect_cheating

cheating_message = "All clear"


app = Flask(__name__)

camera = cv2.VideoCapture(0)  # 0 for webcam

def gen_frames():
    global cheating_message
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, cheating_event = detect_cheating(frame)
            cheating_message = cheating_event if cheating_event else "All clear"

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def alerts():
    def event_stream():
        global cheating_message
        prev_message = ""
        while True:
            if cheating_message != prev_message:
                yield f"data: {cheating_message}\n\n"
                prev_message = cheating_message
    return Response(event_stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
