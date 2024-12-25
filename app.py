from flask import Flask, render_template, request, redirect, url_for
from src.citra import predict
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("predict.html", error="No file selected")

        file = request.files["file"]
        if file.filename == "":
            return render_template("predict.html", error="No file selected")

        # Simpan file di folder static/images
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        try:
            class_name, category, confidence = predict(file_path)
            # Hanya simpan nama file, bukan path lengkap
            return render_template(
                "predict.html",
                uploaded_image=f"images/{file.filename}",
                class_name=class_name,
                category=category,
                confidence=confidence,
            )
        except Exception as e:
            return render_template("predict.html", error=f"Prediction error: {str(e)}")

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

