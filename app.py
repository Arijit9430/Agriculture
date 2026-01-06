import os
import json
import numpy as np
import gradio as gr
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGE_SIZE = (128, 128)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "cropinailab/aksara_v1"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

PLANTS = [
    "apple", "tomato", "potato", "orange", "chili", "grape", "tea",
    "peach", "coffee", "corn", "cucumber", "jamun", "lemon",
    "mango", "pepper", "rice", "soybean", "sugarcane", "wheat"
]

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Spanish": "es",
    "French": "fr",
    "German": "de"
}

# ---------------- LOAD CLASS MAPS ----------------
inv_maps = {}

for plant in PLANTS:
    class_map = os.path.join(MODEL_DIR, f"{plant}_class_map.json")
    if os.path.exists(class_map):
        with open(class_map, "r") as f:
            inv_maps[plant] = {v: k for k, v in json.load(f).items()}
    else:
        inv_maps[plant] = {}

# ---------------- MODEL CACHE ----------------
MODEL_CACHE = {}

def load_plant_model(plant):
    if plant not in MODEL_CACHE:
        path = os.path.join(MODEL_DIR, f"{plant}.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model missing: {plant}")
        MODEL_CACHE[plant] = load_model(path, compile=False)
    return MODEL_CACHE[plant]

# ---------------- TRANSLATION ----------------
def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

# ---------------- LLM VIA HF API ----------------
def generate_prevention_llm(plant, disease):
    if not HF_API_TOKEN:
        return "⚠️ Hugging Face API token not found."

    prompt = f"""
You are an agricultural expert specializing in plant pathology, crop nutrition, and safe farm management.
Your job is to provide accurate, scientifically correct, and legally safe advice.

Plant: {plant}
Issue: {disease}

Your response MUST follow this structure clearly and must be 100% accurate:

### 1. About the Disease
- Explain what the disease is and identify the correct pathogen type (fungus, bacteria, virus, pest, oomycete, etc.)
- Describe how it spreads (only scientifically correct modes of spread)
- Avoid any incorrect or exaggerated claims

### 2. Symptoms
- Describe accurate symptoms on each relevant plant part:
  - Leaves
  - Stems
  - Roots
  - Fruit (only if applicable)
  - Tubers/roots if root-based crop

### 3. Safe & Legal Treatment Options
Provide ONLY safe, standard treatments used by agricultural extension services.
Include copper fungicides, mancozeb, chlorothalonil, sulfur (if relevant),
biological controls, and cultural practices.
Never provide dosages.

### 4. Prevention
Include resistant varieties, crop rotation, spacing, airflow,
drip irrigation, sanitation, and monitoring.

### 5. Nutrient Requirements
Explain N, P, K, Ca, Mg, S and micronutrients roles.

### 6. Fertilizer Recommendations (No Dosages)
Chemical, organic, and biofertilizers with explanation.

### 7. Additional Good Practices
Irrigation, drainage, sanitation, hygiene, storage.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        output = response.json()

        if isinstance(output, list):
            return output[0]["generated_text"].replace(prompt, "").strip()

        return "No response from LLM."

    except Exception as e:
        return f"LLM Error: {e}"

# ---------------- PREDICTION ----------------
def predict(image_input, plant, language):
    if image_input is None:
        return "❌ Please upload an image.", ""

    model = load_plant_model(plant)

    img = image_input.convert("RGB").resize(IMAGE_SIZE)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    disease = inv_maps.get(plant, {}).get(idx, "Unknown Disease")

    prevention = generate_prevention_llm(
        plant.capitalize(),
        disease.replace("_", " ")
    )

    prevention = translate_text(prevention, language)

    result = f"""
### 🌿 Detected Disease
**{disease.replace("_", " ")}**

### 📊 Confidence
**{confidence:.2%}**

### 🛡️ Prevention & Cure (AI Generated)
{prevention}
"""

    return result, f"{confidence:.2%}"

# ---------------- GRADIO UI ----------------
with gr.Blocks(title="🌱 Plant Disease Detection") as demo:
    gr.Markdown("# 🌱 Plant Disease Detection")
    gr.Markdown("Upload a leaf image to detect disease and get AI-based prevention advice.")

    plant = gr.Dropdown(PLANTS, label="Select Plant", value="apple")
    language = gr.Dropdown(list(LANGUAGES.keys()), value="English", label="Select Language")
    image_input = gr.Image(type="pil", label="Upload Leaf Image")

    detect_btn = gr.Button("Detect Disease", variant="primary")

    output_md = gr.Markdown()
    confidence_txt = gr.Textbox(label="Confidence", interactive=False)

    detect_btn.click(
        fn=lambda img, p, lang: predict(img, p, LANGUAGES[lang]),
        inputs=[image_input, plant, language],
        outputs=[output_md, confidence_txt]
    )

demo.launch()
