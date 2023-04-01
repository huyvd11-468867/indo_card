from main_card import MainCard
from random import choice
from flask import Flask, jsonify, request
import requests
from PIL import Image
from io import BytesIO
import uuid
import os
from flask_cors import CORS, cross_origin
from functools import lru_cache
import socket
import base64
from config import *

model_card_general = MainCard()

for fd in list_folder_save:
    if not os.path.exists(fd):
        os.mkdir(fd)


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


desktop_agents = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


def random_headers():
    return {'User-Agent': choice(desktop_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}


def download_image(image_url):
    header = random_headers()
    response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image


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


def create_response_table(data, error_code, error_message):
    response = {
        "data": data,
        "errorCode": error_code,
        "errorMessage": error_message,
    }
    return response


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@lru_cache(maxsize=1)
@app.route("/card", methods=['GET', 'POST'])
@cross_origin()
def card():
    delete_file = request.args.get('delete_file', default='false', type=str)
    get_thumb = request.args.get('get_thumb', default='false', type=str)
    format_type = request.args.get('format_type', default="url", type=str)
    if request.method == "POST":
        if format_type not in params_post:
            return jsonify_str(create_response("", "", "6", "Incorrect format type"))
        if format_type == "file":
            try:
                pdf_data = request.files["img"]
                name_save_pdf = os.path.join(folder_save_card, uuid.uuid4().hex + ".pdf")
                pdf_data.save(name_save_pdf)
            except Exception as ex:
                return jsonify_str(create_response("", "", "3", "Incorrect image format"))
        else:
            try:
                content = request.get_json()
                image_encode = content['img']
                img = base64.b64decode(str(image_encode))
                name_save_pdf = os.path.join(folder_save_card, uuid.uuid4().hex + ".pdf")
                with open(name_save_pdf, "wb") as f:
                    f.write(img)
            except Exception as ex:
                return jsonify_str(create_response("", "", "3", "Incorrect image format"))
    else:
        if format_type not in params_get:
            return jsonify_str(create_response("", "", "6", "Incorrect format type"))
        try:
            image_url = request.args.get('img', default='', type=str)
            img = download_image(image_url)
            name_save_pdf = os.path.join(folder_save_card, uuid.uuid4().hex + ".pdf")
            img.save(name_save_pdf)
        except Exception as ex:
            return jsonify_str(create_response("", "", "2", "Url is unavailable"))
    result_total = model_card_general.predict_card_general(name_save_pdf)
    if delete_file.lower() == "true":
        os.remove(name_save_pdf)
    if len(result_total) == 0:
        return jsonify_str(create_response("", "", "1", "The photo does not contain content"))
    if get_thumb.lower() == "false":
        for i, item in enumerate(result_total):
            result_total[i]['img'] = ''
    return jsonify_str(create_response_table(result_total, "0", "Success"))


@lru_cache(maxsize=1)
@app.route("/credit_card", methods=['GET', 'POST'])
@cross_origin()
def credit_card():
    get_thumb = request.args.get('get_thumb', default='false', type=str)
    format_type = request.args.get('format_type', default="url", type=str)
    if request.method == "POST":
        if format_type not in params_post:
            return jsonify_str(create_response("", "", "6", "Incorrect format type"))
        if format_type == "file":
            try:
                pdf_data = request.files["img"]
                name_save_pdf = os.path.join(folder_save_credit_card, uuid.uuid4().hex + ".pdf")
                pdf_data.save(name_save_pdf)
            except Exception as ex:
                return jsonify_str(create_response("", "", "3", "Incorrect image format"))
        else:
            try:
                content = request.get_json()
                image_encode = content['img']
                img = base64.b64decode(str(image_encode))
                name_save_pdf = os.path.join(folder_save_credit_card, uuid.uuid4().hex + ".pdf")
                with open(name_save_pdf, "wb") as f:
                    f.write(img)
            except Exception as ex:
                return jsonify_str(create_response("", "", "3", "Incorrect image format"))
    else:
        if format_type not in params_get:
            return jsonify_str(create_response("", "", "6", "Incorrect format type"))
        try:
            image_url = request.args.get('img', default='', type=str)
            img = download_image(image_url)
            name_save_pdf = os.path.join(folder_save_credit_card, uuid.uuid4().hex + ".pdf")
            img.save(name_save_pdf)
        except Exception as ex:
            return jsonify_str(create_response("", "", "2", "Url is unavailable"))
    result_total = model_card_general.predict_credit_cards(name_save_pdf)
    if len(result_total) == 0:
        return jsonify_str(create_response("", "", "1", "The photo does not contain content"))
    if get_thumb.lower() == "false":
        for i, item in enumerate(result_total):
            result_total[i]["info"]['image'] = ''
    return jsonify_str(create_response_table(result_total, "0", "Success"))


app.run(get_ip(), port=port_service, threaded=False, debug=False)
