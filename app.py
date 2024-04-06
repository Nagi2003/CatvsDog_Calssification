from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io

app = Flask(__name__)

def load_and_preprocess_image(image_file):
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image

def predict_image_class(model, image):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    if prediction > 0.5:
        return "Cat"
    else:
        return "Dog"

model = load_model("model.h5")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = load_and_preprocess_image(file)
            print("Image shape:", image.shape)  # Debug statement
            prediction = predict_image_class(model, image)
            print("Prediction:", prediction)  # Debug statement

            # Convert image to base64 for passing to template
            image_buffer = io.BytesIO()
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image_pil.save(image_buffer, format='JPEG')
            image_base64 = base64.b64encode(image_buffer.getvalue()).decode()
            return render_template('result.html', prediction=prediction, image_base64=image_base64)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
