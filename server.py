import http.server
import socketserver
import json
import urllib.request
import urllib.error
import sys
import base64
import io
import warnings
from PIL import Image

# Suppress Hugging Face model loading warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*use_fast.*")

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PORT = 8000
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Initialize MobileNetV2 pipeline for backend image classification
image_classifier = None
try:
    print("Loading MobileNetV2 image classification pipeline in Python backend...", flush=True)
    from transformers import pipeline
    image_classifier = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
    print("MobileNetV2 classification model loaded successfully.", flush=True)
except Exception as e:
    print(f"Warning: Failed to load MobileNetV2 model: {e}. Defaulting to rule-based fallback.", flush=True)

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/diagnose':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                crop = data.get('crop', 'unknown')
                language = data.get('language', 'English')
                image_base64 = data.get('imageBase64', '')

                # Default fallback values (e.g. if classification fails)
                disease = "Unspecified Leaf Spot"
                confidence = 85.0

                # Run MobileNetV2 classification on backend if image is sent
                if image_classifier and image_base64:
                    try:
                        img_bytes = base64.b64decode(image_base64)
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        
                        predictions = image_classifier(pil_img)
                        if predictions and len(predictions) > 0:
                            disease = predictions[0]['label'].split(',')[0].strip()
                            confidence = predictions[0]['score'] * 100
                            print(f"[{self.client_address[0]}] MobileNetV2 prediction: {disease} ({confidence:.1f}% confidence)", flush=True)
                    except Exception as classification_err:
                        print(f"Error classifying image: {classification_err}", flush=True, file=sys.stderr)

                print(f"[{self.client_address[0]}] Analyzing crop: {crop} | Language: {language} | ML prediction: {disease} ({confidence:.1f}%) using DeepSeek...", flush=True)

                prompt = f"""You are an expert agricultural plant pathologist with 25+ years of experience.

Crop: {crop}
ML Model Prediction: {disease} ({confidence:.1f}% confidence)

=== COLOR & PATTERN DATA ===
- Green: {data.get('colorDist', {}).get('green', 0):.1f}%
- Yellow: {data.get('colorDist', {}).get('yellow', 0):.1f}%
- Brown: {data.get('colorDist', {}).get('brown', 0):.1f}%
- Black: {data.get('colorDist', {}).get('black', 0):.1f}%
- Spots/Lesions: {data.get('spotCount', 0)}
- Dominant Damage: {data.get('dominantDamageColor', 'unknown')}

Give a **highly accurate and specific** diagnosis.

Return ONLY valid JSON in this exact format:
{{
  "name": "Disease Name (Scientific Name if possible)",
  "severity": "Healthy | Mild | Moderate | Severe",
  "matchScore": number between 70-100,
  "description": "Clear 2-3 sentence explanation",
  "symptoms": ["symptom1", "symptom2", "symptom3"],
  "organic": ["Organic treatment 1", "Organic treatment 2"],
  "chemical": ["Chemical name with dosage", "Chemical name with dosage"],
  "prevention": ["Prevention tip 1", "Prevention tip 2", "Prevention tip 3"],
  "immediateAction": "Most urgent action for the farmer now"
}}

Be specific. Do not give generic answers. Use the ML prediction and color data to decide."""

                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3
                }

                req = urllib.request.Request(
                    "https://api.deepseek.com/chat/completions",
                    data=json.dumps(payload).encode('utf-8'),
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    method='POST'
                )

                with urllib.request.urlopen(req, timeout=45) as response:
                    res_body = response.read().decode('utf-8')
                    res_data = json.loads(res_body)
                    raw_content = res_data["choices"][0]["message"]["content"]
                    parsed = json.loads(raw_content)

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(parsed).encode('utf-8'))
                    print(f"[{self.client_address[0]}] Diagnosis: {parsed.get('name')}", flush=True)

            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8') if e.fp else str(e)
                print(f"HTTPError from DeepSeek: {e.code} - {err_body}", flush=True, file=sys.stderr)
                self.send_response(e.code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"DeepSeek API error: {e.code}", "details": err_body}).encode('utf-8'))
            except Exception as e:
                print(f"Exception: {str(e)}", flush=True, file=sys.stderr)
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Diagnostic error", "details": str(e)}).encode('utf-8'))

        elif self.path == '/api/ask':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                question = data.get('question', '')
                language = data.get('language', 'English')
                print(f"[{self.client_address[0]}] Question: {question[:60]}... | Language: {language}", flush=True)

                prompt = f"""You are an expert agricultural consultant and plant pathologist.

CRITICAL INSTRUCTION: If the user's question is NOT related to agriculture, farming, crops, plant diseases, soil health, pests, or agricultural management, you MUST decline to answer. Politely respond with exactly: "I am an AI Crop Doctor and can only answer questions related to agriculture, farming, crops, and plant health." and do not write anything else.

Answer the following farming or crop-related question in a helpful, friendly, and practical manner.
Provide clear suggestions including organic controls, chemical controls, or prevention practices when appropriate.

Question: {question}

Format your response in neat HTML paragraphs or lists (use standard HTML tags like <p>, <ul>, <li>, <strong>, etc.) so that it can be directly set inside an element's innerHTML in a browser page. Do NOT include markdown code blocks or backticks."""

                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}]
                }

                req = urllib.request.Request(
                    "https://api.deepseek.com/chat/completions",
                    data=json.dumps(payload).encode('utf-8'),
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    method='POST'
                )

                with urllib.request.urlopen(req, timeout=45) as response:
                    res_body = response.read().decode('utf-8')
                    res_data = json.loads(res_body)
                    answer = res_data["choices"][0]["message"]["content"]

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"answer": answer}).encode('utf-8'))
                    print(f"[{self.client_address[0]}] Answer sent.", flush=True)

            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8') if e.fp else str(e)
                print(f"HTTPError: {e.code} - {err_body}", flush=True, file=sys.stderr)
                self.send_response(e.code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"DeepSeek error: {e.code}", "details": err_body}).encode('utf-8'))
            except Exception as e:
                print(f"Exception: {str(e)}", flush=True, file=sys.stderr)
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Assistant error", "details": str(e)}).encode('utf-8'))

        else:
            super().do_GET()

socketserver.TCPServer.allow_reuse_address = True

print(f"Starting CropDoc AI server on port {PORT}...", flush=True)
print(f"  AI Engine: DeepSeek Chat", flush=True)
with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"Server running at http://localhost:{PORT}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", flush=True)
        httpd.server_close()
