import json
from flask import Flask, flash, redirect, request
import PIL
from model import Detection, Recognition

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

detection = Detection()
recognition = Recognition()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ocr_model_run(image, detection, recognition):
    detection_result = detection.predict(image)
    recognition_result = recognition.predict(detection_result, image)
    return recognition_result

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict_image', methods=['POST', 'GET'])
def predict_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected image')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            image = PIL.Image.open(file.stream)
            receipt_data = ocr_model_run(image, detection, recognition)
            json_data = json.dumps(receipt_data)
            return json_data


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=80)
