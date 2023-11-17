import cv2
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from DeepStream.DeepStream import pipeline

''' 
                                                ! WARNING !
    Currently this Application is not used by Deepstream application. It will be featured in next versions
'''


class VideoCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.pipeline = pipeline
        self.fps = 0
        self.start_time = time.time()
    def run(self):
        while self.running:
            ret, frame = self.pipeline.get_frame()
            if ret:
                frame, fps = self.process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    self.frame = buffer.tobytes()
                    self.fps = fps

    def stop(self):
        self.running = False
        self.capture.release()

    def process_frame(self, frame):
        ##### FACE DETECTION #####
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ##### END OF FACE DETECTION #####

        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        fps = int(1 / elapsed_time)
        self.start_time = time.time()

        # Overlay FPS on the frame
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame, fps

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    frame = video_thread.frame
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                print('Video stream closed:', str(e))
        elif self.path == '/styles.css':
            self.send_response(200)
            self.send_header('Content-type', 'text/css')
            self.end_headers()
            with open('styles.css', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())


if __name__ == '__main__':
    video_thread = VideoCaptureThread()
    video_thread.start()

    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Server started on http://localhost:8000')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        video_thread.stop()
        video_thread.join()
        httpd.server_close()
        print('Server stopped')