# Face Mask & Plant Disease Detection

An AI-powered web application that detects:
- 😷 Whether a person is **wearing a face mask or not**
- 🌿 Whether a plant leaf is **healthy or diseased**

Powered by **Groq API** (Llama 4 Scout vision model) for fast and accurate predictions.

## Tech Stack

| Layer    | Technology |
|----------|-----------|
| Backend  | FastAPI + Python |
| AI       | Groq API — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Database | SQLite (local) |
| Frontend | Vanilla HTML/CSS/JS |
| Hosting  | Render |

## Project Structure

```
project/
├── backend/
│   ├── main.py          # FastAPI app, routes
│   ├── database.py      # SQLAlchemy DB setup
│   └── models.py        # DB models
├── frontend/
│   ├── index.html       # Home / navigation
│   ├── upload.html      # Upload & scan page
│   ├── result.html      # Prediction result page
│   ├── admin.html       # Admin dashboard
│   ├── about.html
│   └── contact.html
├── src/
│   ├── model.py         # MobileNetV2 model definition
│   ├── train.py         # Training script
│   ├── predict.py       # Inference class
│   ├── evaluate.py      # Evaluation script
│   └── data_preprocessing.py
├── utils/
│   └── helper.py
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── render.yaml          # Render deployment config
└── .env.example         # Environment variable template
```

## Local Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/NARALAMANOJYADAV/face-mask-and-plant-disease-detection.git
   cd face-mask-and-plant-disease-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   ```

4. Run the server:
   ```bash
   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8080
   ```

5. Open: http://localhost:8080/ui/

## Getting a Groq API Key

1. Go to https://console.groq.com
2. Sign up / log in
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key and paste it into your `.env` file as `GROQ_API_KEY=...`

## Deploying to Render

See **DEPLOY.md** for full step-by-step instructions.
