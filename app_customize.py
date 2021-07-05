from flask_cors import CORS
import configparser
import json
import sys
from logging import getLogger, basicConfig, INFO
import os
import cv2
from flask import Flask, Response, render_template, request, redirect, jsonify
from flask_cors import CORS, cross_origin

from libs.argparser import build_argparser
from libs.camera import VideoCamera
from libs.interactive_detection import Detections

app = Flask(__name__, static_folder="./static", template_folder="./templates")
CORS(app) #Cross Origin Resource Sharing
logger = getLogger(__name__)

basicConfig(
    level=INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s"
)

# config = configparser.ConfigParser()
# config.read("config.ini")

# detection control flag
is_async = True
is_det = True
is_reid = True
# 0:x-axis 1:y-axis -1:both axis
flip_code = 1
resize_width = 640

UPLOAD_FOLDER = './file/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def gen(camera, model, input_src):
    if not camera:
       camera = VideoCamera(input_src, resize_width, False)

    while True:
        frame = camera.get_frame(flip_code)
        if model == 'default':
            frame = detections.person_detection(frame, is_async, is_det, is_reid)
        elif model == 'csr':
            pass # TODO
        else:
            return

        ret, jpeg = cv2.imencode(".jpg", frame)
        frame = jpeg.tostring()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/")
@cross_origin()
def index():
    return render_template(
        "index.html", is_async=is_async, flip_code=flip_code, enumerate=enumerate,
    )


@app.route("/video_feed")
@cross_origin()
def video_feed():
    model = request.args.get('model')
    input_src = request.args.get('input')
    model = model if model else "default" # default vs csr
    input_src = input_src if input_src else "cam" # cam or file_name

    return Response(gen(camera, model, input_src), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        # Remove existing images in directory
        files_in_dir = os.listdir(app.config['UPLOAD_FOLDER'])
        filtered_files = [file for file in files_in_dir if file.endswith(".mp4")]
        for file in filtered_files:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            os.remove(path)

        # Upload new file
        if 'file' not in request.files:
            return "error"
        file = request.files['file']

        if not file:
            return "error"

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return "done"


if __name__ == "__main__":
    # arg parse
    args = build_argparser().parse_args()
    devices = [args.device, args.device_reidentification]

    camera = VideoCamera(args.input, resize_width, args.v4l)
    logger.info(
        f"input:{args.input} v4l:{args.v4l} frame shape: {camera.frame.shape} axis:{args.axis} grid:{args.grid}"
    )
    detections = Detections(camera.frame, devices, args.axis, args.grid)

    app.run(host="0.0.0.0", threaded=True)
    # app -i cam
    # app -i people-waling.mp4 --grid 10
