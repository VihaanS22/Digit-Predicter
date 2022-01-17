from flask import Flask, jsonify, request
from model import prediction

app = Flask(__name__)
@app.route("/digit-pred", methods = ["POST"])
def predict_digit():
    image = request.files.get("digit")
    pred = prediction(image)
    return jsonify({
        "Prediction of the digit shown" : pred
    }), 200

if __name__ == "__main__":
    app.run(debug = True)


