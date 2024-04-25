from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        model = YOLO(f'runs/segment/train/weights/best.pt')
        results = model(os.path.join(app.config['UPLOAD_FOLDER'], filename), stream=True)
        
        # Process each frame in the results
        for result in results:
            # Plot results image
            im_bgr = result.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            
            # Create the directory if it doesn't exist
            os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
            
            # Save results to disk
            result_filename = f"{filename}_result.jpg"
            im_rgb.save(os.path.join(app.config['RESULT_FOLDER'], result_filename))
            
            # return results to user
            return render_template('display.html', original=filename, result=result_filename)
    
    return '''
    <!doctype html>
    <title>Upload an image</title>
    <h1>Upload an image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def send_result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8081)