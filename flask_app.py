from flask import Flask, request, jsonify
from fer import FER
import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
import json

# Flask 애플리케이션 생성 코드
app = Flask(__name__)

# 현재 스크립트 파일의 절대 경로를 가져옴
script_dir = os.path.dirname(os.path.abspath(__file__))

# 로그 기록을 위한 코드
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = os.path.join(script_dir, 'app.log')
file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=10)  # Save up to 10 MB of log files
file_handler.setLevel(logging.DEBUG)  # Set log level to DEBUG
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)

# 애플리케이션에서 발생하는 모든 예외를 처리하는 함수
@app.errorhandler(Exception)
def handle_error(e):
    # Print the traceback for detailed error information
    traceback.print_exc()

    # Log the error message
    app.logger.error(f"An error occurred: {str(e)}")

    # Return a JSON response with an error message
    return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Post 메서드로 '/auth/detect-emotions' 엔드포인틀르 생성
@app.route('/auth/detect-emotions', methods=['POST'])
def detect_emotions():
    try:
        # Print the request data for debugging
        app.logger.info("Request data: %s", request.files)

        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        # 클라이언트로부터 전송된 파일을 읽음
        file = request.files['file']  # Accessing the file using the key 'file'

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # OpenCV를 사용하여 이미지를 디코딩함
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Initialize the FER detector
        detector = FER()

        # Detect emotions in the image
        result = detector.detect_emotions(img)

        # Serialize ndarray objects
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Serialize result
        result_serialized = json.dumps(result, default=serialize)

        # Log the result
        app.logger.info("Emotion detection result: %s", result_serialized)

        # Print the result to the console
        print("Emotion detection result:", result_serialized)

        # Return the serialized result as JSON
        return jsonify(json.loads(result_serialized))
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Create the log file if it doesn't exist
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
    
    app.run(debug=True, port=5001)
