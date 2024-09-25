import cv2
import numpy as np
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, Quality
from picamera2.outputs import FileOutput
from libcamera import controls, Transform
from threading import Condition
from io import BufferedIOBase
import json

app = Flask(__name__)

# 載入Haar級聯分類器模型進行人臉檢測
haar_cascade_path = '/home/jimmy/camera-opencv/camera-opencv/03-face_datection/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# 載入訓練好的LBPH人臉識別模型
model_path = '/home/jimmy/camera-opencv/camera-opencv/03-face_datection/models/trained_face_recognition_model.yml'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)

# 載入標籤字典
label_dict_path = '/home/jimmy/camera-opencv/camera-opencv/03-face_datection/models/label_dict.json'
with open(label_dict_path, 'r', encoding='utf-8') as f:
    label_dict = json.load(f)

# 創建一個Picamera2
cam = Picamera2()

# 調整相機
config = cam.create_video_configuration(
    main={'size': (1280, 720), 'format': 'XBGR8888'}, # 調整分辨率
    transform=Transform(vflip=1),
    controls={
        'NoiseReductionMode': controls.draft.NoiseReductionModeEnum.HighQuality,
        'Sharpness': 1.5,
        #幀率
        'FrameRate': 3 
    }
)

class StreamingOutput(BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
              # 解碼幀進行人臉檢測
            image_array = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

            # 進行人臉檢測
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # 增加圖片對比度
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # scaleFactor(越大越快、但會有疏失)就是縮放比例
                minNeighbors=5,   # minNeighbors 檢測次數
                minSize=(50, 50)  # 檢測尺寸
            )

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                label, confidence = face_recognizer.predict(face)
                if confidence < 80:  #可信度，很像找出來的匹配率標準，越小越像，所以小於此值才會顯示
                    name = label_dict.get(str(label), "Unknown")
                    text = f"{name}: {confidence:.2f}"
                    color = (0, 255, 0)  # 設定方框為綠色
                else:
                    text = "Unknown"
                    color = (0, 0, 255)  # 設定方框為紅色

                cv2.rectangle(image_array, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image_array, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 重新編碼幀為JPEG
            _, jpeg = cv2.imencode('.jpg', image_array)
            self.frame = jpeg.tobytes()
            self.condition.notify_all()

output = StreamingOutput()


cam.configure(config)


cam.start_recording(JpegEncoder(), FileOutput(output), Quality.VERY_HIGH)

@app.route("/")
def index():
    return render_template("index.html")

def gen_frames():
    while True:
        with output.condition:
            output.condition.wait()
            frame = output.frame
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/api/stream')
def video_stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Stop the camera recording after the Flask server ends
cam.stop_recording()
