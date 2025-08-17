from flask import Blueprint, request, jsonify
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # reduce RAM usage


from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os
import io
from tensorflow.image import resize as tf_resize
import requests
import shutil
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download


# Go to project root (flask_of_ml/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models from the "models" folder at root
model_path = hf_hub_download(
    repo_id="RiteshPx/my_models",  # your repo
    filename="final_yet_85.h5"     # uploaded file name
)
model = load_model(model_path)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

userVerify_bp = Blueprint("userVerifyDetails", __name__)


def convert_to_ela_image(image_path, quality=90):
    """
    Converts an image to its Error Level Analysis (ELA) representation.
    
    This function now uses an in-memory buffer to avoid disk conflicts
    during parallel processing.
    """
    # Open the image, convert to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Create an in-memory buffer
    buffer = io.BytesIO()
    
    # Save the image to the buffer at the specified quality
    image.save(buffer, 'JPEG', quality=quality)
    
    # Rewind the buffer to the beginning and open it as a temporary image
    buffer.seek(0)
    temp_image = Image.open(buffer)
    
    # Calculate the difference between the original and re-saved image
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # The buffer will be automatically garbage-collected
    return ela_image

def predict_image(image_path, model, img_size=(128, 128)):        
        """
        Predicts if a new image is real or tampered.
        """
        try:
            # Pre-process the new image
            ela_img = convert_to_ela_image(image_path)
            ela_img = ela_img.resize(img_size)
            img_array = np.array(ela_img)
            img_array = img_array.astype('float32') / 255.0
            
            # The model expects a batch, so we add an extra dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get the prediction
            prediction = model.predict(img_array)
            
            # Get the predicted class (0 or 1)
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            # Get the confidence score
            confidence = prediction[0][predicted_class]
            
            if predicted_class == 0:
                result = "Authentic"
            else:
                result = "Tampered"
            
            print(f"\nPrediction for {image_path}: {result} with confidence {confidence:.4f}")
            return predicted_class
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            


@userVerify_bp.route("/verifyDocuments", methods=["POST"])
def verify_documents():
    data = request.get_json(force=True)
    image_links = data.get("Evidences_link", [])
    if not image_links:
        return jsonify({"error": "Evidences_link is required"}), 400
    if not isinstance(image_links, list):
        return jsonify({"error": "Evidences_link must be a list"}), 400

    local_paths = []
    results = []
    try:
        # Download all images to local paths
        for i, image_link in enumerate(image_links):
            response = requests.get(image_link, stream=True)
            if response.status_code == 200:
                parsed_url = urlparse(image_link)
                filename = os.path.basename(parsed_url.path)
                if not filename:
                    filename = f"temp_image_{i}.jpg"
                local_path = os.path.join(BASE_DIR, "temp_" + filename)
                with open(local_path, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                local_paths.append(local_path)
            else:
                return jsonify({"error": f"Failed to download image from {image_link}"}), 400

        # Predict and clean up
        for local_path in local_paths:
            try:
                print(f"Processing {local_path}...")
                ext = os.path.splitext(local_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    res = predict_image(local_path, model)
                    results.append(res)
                else:
                    results.append(0)
            finally:
                try:
                    os.remove(local_path)
                    print(f"Deleted {local_path}")
                    results = np.array(results).tolist()
                except Exception as e:
                    print(f"Warning: Could not delete {local_path}: {e}")

        return jsonify({
            "message": "Document(s) verified successfully",
            "response": results
        }), 200

    except Exception as e:
        # Clean up any downloaded files if error occurs
        for local_path in local_paths:
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    print(f"Deleted {local_path} due to error")
                except Exception:
                    pass
        return jsonify({"error": str(e)}), 500
