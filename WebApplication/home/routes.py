# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import os
import uuid
import torch
import cv2
from flask import render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from apps.home import blueprint
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from werkzeug.utils import secure_filename
import uuid
from flask import current_app
import sys
from flask import jsonify
import sys
import os

import sys
import os
import importlib.util

#custom_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../apps/ultralytics/ultralytics/nn/modules'))
#print("Files in modules directory:", os.listdir(custom_module_path))
#mycbam_path = os.path.join(custom_module_path, 'mycbam.py')

#spec = importlib.util.spec_from_file_location("mycbam", mycbam_path)
#mycbam = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(mycbam)
from ultralytics.nn.modules import mycbam
print(mycbam)


MODEL_METRICS = {
    "pearbags_cbam.pt": {
        "name": "YOLOv8-CBAM",
        "Precision": "96.6%",
        "Recall": "97.5%",
        "mAP50": "99.0%",
        "mAP50-95": "76.0%"
    },
    "pearbags_cbam_bifpn.pt": {
        "name": "YOLOv8-CBAM-BiFPN",
        "Precision": "95.5%",
        "Recall": "97.6%",
        "mAP50": "98.6%",
        "mAP50-95": "76.0%"
    },
    "pearbags_yolov8.pt": {
        "name": "YOLOv8",
        "Precision": "96.8%",
        "Recall": "98.3%",
        "mAP50": "99.1%",
        "mAP50-95": "74.9%"
    },
    "pearbags_yolov11.pt": {
        "name": "YOLOv11",
        "Precision": "96.8%",
        "Recall": "98.1%",
        "mAP50": "98.9%",
        "mAP50-95": "73.9%"
    },
}



@blueprint.route('/model_metrics', methods=['POST'])
def model_metrics():
    data = request.get_json()
    model_name = data.get('model')
    return jsonify(MODEL_METRICS.get(model_name, {}))


@blueprint.route('/index')
@login_required
def index():

    return render_template('home/index.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


@blueprint.route('/detect', methods=['POST'])
@login_required
def detect():

    model_file = request.form['model']
    video_file = request.files['video']

    # Paths
    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
    output_folder = os.path.join(current_app.root_path, 'static', 'results')
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Save uploaded video
    video_filename = secure_filename(video_file.filename)
    upload_path = os.path.join(upload_folder, video_filename)
    video_file.save(upload_path)

    # Model path
    model_path = os.path.join(current_app.root_path, 'models', model_file)

    # Output video path
    output_filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(output_folder, output_filename)

    # Run detection
    count = run_detection(model_path, upload_path, output_path)

    # Convert to browser-compatible format
    compatible_output_path = convert_to_browser_compatible(output_path)
    output_filename = os.path.basename(compatible_output_path)

    model_metrics = MODEL_METRICS.get(model_file, {})
    model_name = model_metrics.get("name", model_file.replace(".pt", "").upper())

    # Pass relative filename to template
    return render_template('home/index.html', video_result=output_filename, count=count, selected_models=[model_name],model_metrics={model_name: model_metrics})

import subprocess

def convert_to_browser_compatible(input_path):
    base, _ = os.path.splitext(input_path)
    output_path = base + "_web.mp4"
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite without asking
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path


def run_detection(model_path, input_video, output_video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    tracker = DeepSort(max_age=30, max_iou_distance=0.7)

    seen_ids = set()
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    skip_rate = 3  # Process every 3rd frame
    frame_num = 0
    while True:
       ret, frame = cap.read()
       if not ret:

        break
       frame_num += 1
       if frame_num % skip_rate != 0:
        continue
       # Inference
       results = model.predict(source=frame, device=device, verbose=False)
       detections = []

       for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box

        # Confidence threshold
        if conf < 0.6:

            continue

        # Class filtering (assuming pearbag class is 0)
        if int(cls) != 0:

            continue

        # Size and aspect ratio filtering
        w_box, h_box = x2 - x1, y2 - y1
        area = w_box * h_box
        aspect_ratio = w_box / h_box if h_box != 0 else 0

        if area < 300 or aspect_ratio > 2.5 or aspect_ratio < 0.3:
            continue

        detections.append(([x1, y1, w_box, h_box], conf, int(cls)))

        # DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
               continue

            track_id = track.track_id
            if track_id not in seen_ids:
               
               seen_ids.add(track_id)

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, f'Total Unique Pearbags: {len(seen_ids)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
        out.write(frame)
    cap.release()
    out.release()
    print("Saved result to:", output_video)

    return len(seen_ids)

# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
