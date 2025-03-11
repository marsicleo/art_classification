import numpy as np
import mlflow.tensorflow
from PIL import Image
import gradio as gr
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
import requests
from io import BytesIO
import json

# -------------------------------------------------------------------------
# 1. Gradio
# -------------------------------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# -------------------------------------------------------------------------
# 2. Učitavanje modela iz MLflow-a
# -------------------------------------------------------------------------

model_name = "Tate_Hierarchical_Model"
model_version = 10
model_uri = f"models:/{model_name}/{model_version}"

loaded_model = mlflow.tensorflow.load_model(model_uri)

# -------------------------------------------------------------------------
# 3. Učitavanje oznaka encoder-a
# -------------------------------------------------------------------------

classes = np.load("label_classes.npy", allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = classes
print(label_encoder.classes_)

def extract_mapping(label_encoder):
    specific_to_broad = {}
    for label in label_encoder.classes_:
        broader_category, specific_medium = label.split(" > ")
        specific_to_broad[specific_medium] = broader_category
    return specific_to_broad

specific_to_broad_mapping = extract_mapping(label_encoder)

# -------------------------------------------------------------------------
# 4. Preprocesiranje (skaliranje i normalizacija)
# -------------------------------------------------------------------------

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((128, 128))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------------------------------------------------------------
# 5. Predikcija za sučelje sa jednom slikom
# -------------------------------------------------------------------------

def classify_image(img: Image.Image) -> dict:
    x = preprocess_image(img)
    predictions = loaded_model.predict(x)
    pred_index = np.argmax(predictions, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([pred_index])[0]
    broader_category, specific_medium = predicted_label.split(" > ")
    return broader_category, specific_medium

def process_csv(csv_file: str) -> (str, str):
    df = pd.read_csv(csv_file)

    if 'thumbnailUrl' not in df.columns or 'medium' not in df.columns:
        raise ValueError("CSV must contain 'thumbnailUrl' and 'medium' columns.")

    results = []
    correct_predictions = 0

    for _, row in df.iterrows():
        url = row['thumbnailUrl']
        true_specific_medium = row['medium']
        true_broad_category = specific_to_broad_mapping.get(true_specific_medium, None)

        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            predicted_broad_category, predicted_specific_medium = classify_image(img)
            is_correct = predicted_broad_category == true_broad_category
            correct_predictions += int(is_correct)
            results.append({
                "thumbnailUrl": url,
                "True Specific Medium": true_specific_medium,
                "True Broader Category": true_broad_category,
                "Predicted Broader Category": predicted_broad_category,
                "Predicted Specific Medium": predicted_specific_medium,
                "Is Correct": is_correct,
                "Error": None
            })

        except Exception as e:
            results.append({
                "thumbnailUrl": url,
                "True Specific Medium": true_specific_medium,
                "True Broader Category": true_broad_category,
                "Predicted Broader Category": None,
                "Predicted Specific Medium": None,
                "Is Correct": False,
                "Error": str(e)
            })

    results_df = pd.DataFrame(results)

    output_path = "predictions.csv"
    results_df.to_csv(output_path, index=False)

    total = len(df)
    accuracy = correct_predictions / total * 100
    accuracy_message = f"Correct Predictions (Broader Category): {correct_predictions}/{total} ({accuracy:.2f}%)"

    return output_path, accuracy_message

# -------------------------------------------------------------------------
# 6. Gradio Sučelje za pojedinacnu sliku
# -------------------------------------------------------------------------

def process_csv_file(file) -> (str, str):
    output_path, accuracy_message = process_csv(file.name)
    return output_path, accuracy_message

single_image_interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Text(label="Broader Category"),
        gr.Text(label="Specific Medium")
    ],
    title="Single Image Classification",
    description="Upload an image to classify its broader category and specific medium."
)

# -------------------------------------------------------------------------
# 7. Gradio Sučelje za CSV datoteku
# -------------------------------------------------------------------------

batch_csv_interface = gr.Interface(
    fn=process_csv_file,
    inputs=gr.File(type="filepath", label="Upload CSV (with 'thumbnailUrl' and 'medium' columns)"),
    outputs=[
        gr.File(label="Download Predictions CSV"),
        gr.Text(label="Prediction Accuracy")
    ],
    title="Batch Image Classification via CSV",
    description="Upload a CSV file containing image URLs in the 'thumbnailUrl' column and the actual mediums in 'medium'. The model will classify each image and provide a CSV with predictions and accuracy stats."
)

# -------------------------------------------------------------------------
# 8. Spajanje oba sučelja
# -------------------------------------------------------------------------

app = gr.TabbedInterface(
    [single_image_interface, batch_csv_interface],
    ["Single Image Classification", "Batch Classification via CSV"]
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
