from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import time
import csv
app = Flask(__name__)
api = Api(app)


def getLocal():
    with open('PredictedData/Local_Pred.csv', newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)
    return row1


def getAuth():
    with open('PredictedData/Auth_Pred.csv', newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)
    return row1


def getActivity():
    with open('PredictedData/Activity_Pred.csv', newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)
    return row1


@app.route('/get', methods=["GET"])
def data():
    return (jsonify({"activity": getActivity(), "user": getAuth(), "location": getLocal()}))


if __name__ == '__main__':
    app.run(debug=True)
