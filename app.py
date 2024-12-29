import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'dng', 'tif', 'pfm', 'webp', 'bmp', 'png', 'tiff', 'mpo', 'jpg', 'jpeg'}

# Load your YOLO model
model_path = 'engine/best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = YOLO(model_path)
print("Model loaded successfully")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform inference with YOLOv8n
        results = model(file_path)

        # Process results and save annotated image
        if results:
            annotated_image_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            draw_annotations(file_path, results, annotated_image_path)

            # Render the result in the template
            return render_template('index.html', filename=filename, result_image=filename)
        else:
            return render_template('index.html', filename=filename, result_image='no_detections.png')

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

def draw_annotations(input_image_path, results, output_image_path):
    image = Image.open(input_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Load a bold font
    font_path = "DejaVuSans-Bold.ttf"  # Path to your bold font file
    font_size = 16
    font = ImageFont.truetype(font_path, font_size)

    # Loop through each result (assuming results is a list of Results objects)
    for result in results:
        # Loop through each box in the result
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # Get the coordinates in (xmin, ymin, xmax, ymax) format
            label = box.cls.item()  # Get the class label index
            score = box.conf.item()  # Get the confidence score
            class_name = result.names[int(label)]  # Get the class name

            # Draw the bounding box
            draw.rectangle(xyxy, outline="red", width=3)
            # Draw the label and confidence score
            draw.text((xyxy[0], xyxy[1]), f"{class_name} {score:.2f}", fill="yellow", font=font)

    # Save the annotated image with high quality
    image.save(output_image_path, quality=95, subsampling=0)

if __name__ == "__main__":
    app.run(debug=True)
