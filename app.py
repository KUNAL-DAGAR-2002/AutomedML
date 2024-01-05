from flask import Flask, request, render_template
from trainer import num_regressor, num_classification, cat_classification, combined_regressor, combined_classification
import os
from modules.logger import logging
import zipfile

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/submit', methods=['GET', 'POST'])
def categorical():
    if request.method == 'POST':
        category = request.form.get('category').lower()
        type_ = request.form.get('type_').lower()
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
        if category == "numerical" and type_ == "regression":
            num_regressor(file_path)
        elif category == "numerical" and type_ == "classification":
            num_classification(file_path)
        elif category == "categorical":
            cat_classification(file_path)
        elif category == "combined" and type_ == "regression":
            combined_regressor(file_path)
        elif category == "combined" and type_ == "classification":
            combined_classification(file_path)
        folder_path = 'static/artifacts'
        zip_filename = 'model.zip'

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for foldername, subfolders, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

        zipf.close()
    return render_template('index.html')

if __name__ == '__main__':
    logging.info("started")
    logging.info(app.run(debug=True))
