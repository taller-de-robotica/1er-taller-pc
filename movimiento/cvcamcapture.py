#!/usr/bin/python3

#pip3 install numpy
#pip3 install opencv-contrib-python ó pip install opencv-python
#pip3 install imutils

# Originales:
# https://pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
# https://raw.githubusercontent.com/raspberrypi/picamera2/main/examples/mjpeg_server.py
# https://snyk.io/advisor/python/imutils/functions/imutils.video.VideoStream

import cv2
import io
import logging
import socketserver
from http import server
from threading import Condition

from imutils.video import VideoStream
import threading
import time


PAGE = """\
<!DOCTYPE html>
<html>
<head>
<title>Demo con OpenCV</title>
</head>
<body>
<h1>Webcam Streaming Demo</h1>
<img src="stream.mjpg" />
</body>
</html>
"""
            
            
class StreamingHandler(server.BaseHTTPRequestHandler):

    def do_POST(self):
        print("Atiende post")
        if self.path == '/command/':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            print("POST body:", body)
            content = '{"command": "forward", "status": "ok"}'.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            print("Intengo de contactar ", self.path)
            self.send_error(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/command/':
            print("GET command")
            self.send_response(200)
            self.send_header('Location', '/command/')
            self.end_headers()
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                global vs
                global jpeg_quality
                while True:
                    frame = vs.read()
                    #(h, w) = frame.shape[:2]
                    #frame = cv2.resize(frame, (w//2, h//2))
                    ret_code, jpg_buffer = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    )
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(jpg_buffer))
                    self.end_headers()
                    self.wfile.write(jpg_buffer)
                    self.wfile.write(b'\r\n')
                    #cv2.imshow('frame', frame)
            except BrokenPipeError as e:
                logging.info('Broken pipe client $s: %s closed conection.',
                             self.client_address, str(e))
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
                raise e
        else:
            self.send_error(404)
            self.end_headers()
            
            
#class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
class StreamingServer(server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, bind_and_activate):
        super().__init__(server_address, bind_and_activate)
        print("Dirección: ", server_address)
          

if __name__ == '__main__':
    #vs = VideoStream(usePiCamera=True).start()
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    jpeg_quality = 95  # 0 to 100, higher is better quality, 95 is cv2 default
    
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        vs.stop()
        #cv2.destroyAllWindows()

