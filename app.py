from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
MODEL_PATH = os.path.join("model", "cotton_disease_model.h5")

print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Ensure the class labels match the model's training order
CLASS_NAMES = [
    "Aphids", "Army Worm", "Bacterial Blight", "Cotton Boll Rot",
    "Green Cotton Boll", "Healthy", "Powdery Mildew", "Target Spot"
]  # Ensure this order matches model training labels

# Image size used during training
IMG_SIZE = (224, 224)

@app.route('/')
def home():
    print("📌 Home page accessed")
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("📥 Received an upload request")
        file = request.files.get('file')

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"✅ File saved: {file_path}")

            # Load and preprocess the image
            img = image.load_img(file_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            # Predict the disease
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)  # Get the highest confidence class
            confidence_score = np.max(predictions) * 100  # Convert to percentage
            predicted_class = CLASS_NAMES[predicted_index]  # Get class label

            print(f"🔍 Predicted: {predicted_class} with {confidence_score:.2f}% confidence")

            return render_template(
                'result.html', 
                file_name=file.filename, 
                result=predicted_class, 
                accuracy=round(confidence_score, 2)
            )

    # Show upload page if GET request
    print("📌 Upload page accessed")
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=True, use_reloader=False)

