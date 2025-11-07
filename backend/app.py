from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
import torch
from scipy.special import softmax
import pickle

# =====================================
# ⚙️ FastAPI app setup
# =====================================
app = FastAPI(title="Telugu Spam SMS Detector (MuRIL)")

# Allow Streamlit frontend
origins = ["http://localhost:8501", "http://127.0.0.1:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 🧩 Load tokenizer and model
# =====================================
MODEL_PATH = "muril_model.pickle"
MODEL_NAME = "google/muril-base-cased"  # the base MuRIL model

try:
    # Load tokenizer normally
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load pickled model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e


# =====================================
# 📨 Request model
# =====================================
class MessageIn(BaseModel):
    text: str


# =====================================
# 🧮 Prediction function
# =====================================
def predict_label(text: str):
    if not text.strip():
        raise ValueError("Empty text provided.")

    # Clean and tokenize
    cleaned = " ".join(text.split())
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0].detach().cpu().numpy()
        probs = softmax(logits)
        pred = int(probs.argmax())
        label = "Spam" if pred == 1 else "Ham"
        confidence = float(probs[pred])

    return label, confidence


# =====================================
# 🚀 Prediction endpoint
# =====================================
@app.post("/predict")
def predict(payload: MessageIn):
    try:
        label, conf = predict_label(payload.text)
        return {"label": label, "confidence": conf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
