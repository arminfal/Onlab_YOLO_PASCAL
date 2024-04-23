from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        
        model = YOLO(f'/runs/segment/train/weights/best.pt')
        results = model(os.path.join('uploads', filename), stream=True)
        
        # return results to user
        return str(results)
    
    return '''
    <!doctype html>
    <title>Upload an image</title>
    <h1>Upload an image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)