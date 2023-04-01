import os

import requests
from PIL import Image

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
from main import ExtractIndoCard
from config import *

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

# http://127.0.0.1/add
# http://127.0.0.1/minus
# http://127.0.0.1/multi
# http://127.0.0.1/div

model = ExtractIndoCard()

app = Flask(__name__)


def jsonify_str(output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result


def create_response(type_oke, data, error_code, error_message):
    if data == "":
        response = {
            "data": data,
            "errorCode": error_code,
            "errorMessage": error_message,
        }
    else:
        response = {
            "data": data["data"],
            "errorCode": error_code,
            "errorMessage": error_message,
        }
    return response


# Client gửi lên Phương thức GET là phương thức gửi dữ liệu thông qua đường dẫn URL nằm trên thanh địa chỉ của Browser
@app.route('/returnjson', methods=["POST", "GET"])
def returnjson():
    if request.method == "POST":
        try:
            image = request.files['img']
            img_path = os.path.join('static', image.filename)

        except Exception as ex:
            return jsonify_str(create_response("", "", "6", "Incorrect format type"))
    else:
        try:
            img_url = request.values['img']
            img_data = requests.get(img_url).content
            img_path = os.path.join(path_save, 'temp.jpg')
            with open(img_path, 'wb') as f:
                f.write(img_data)

        except Exception as ex:
            return jsonify_str(create_response("", "", "2", "Url is unavailable"))

    img = Image.open(img_path)
    result_total = model.predict(img)
    return jsonify(result_total)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
