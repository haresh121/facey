from flask import Flask, render_template, url_for, request
from model import Model


seq = Model()
seq.save_data()


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/opencam")
def opencam():
    seq.predict_from_cam()
    return """<h2>Ive opend the camera</h2>"""


@app.route("/predict", methods=["POST"])
def predpic():
    img = request.files["photo"]
    names = seq.predict_from_cam(cam=False, pic=True, im=img)
    return render_template("predict.html", names=str(names))


if __name__ == "__main__":
    app.run(debug=True)
