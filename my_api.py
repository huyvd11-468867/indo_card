import json
import os

from PIL import Image

from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from flask import request

from main import ExtractIndoCard

from keras import backend as K

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


@app.route('/returnjson', methods=['POST'])
@cross_origin(origins='*')
def ReturnJSON():
    image = request.files['img']
    if 'img' in request.files:
        img_path = os.path.join('static', image.filename)
        image.save(img_path)

        img = Image.open(img_path)
        data = model.predict(img)

        print(data)
        return jsonify(data)

    else:
        return {}
    # return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='3000')
