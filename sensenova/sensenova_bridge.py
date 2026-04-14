from __future__ import annotations

import ast
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sensenova/SenseNova-SI-1.1-InternVL3-2B"
HOST = "localhost"
PORT = 5050


def log(msg: str) -> None:
    print(msg, flush=True)


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


log("Loading SenseNova Bridge (this can take a moment)...")
DEVICE, DTYPE = get_device_and_dtype()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().to(DEVICE)

log(f"Model loaded on {DEVICE.upper()}. Starting server on port {PORT}...")


def process_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Translates the raw image into the mathematical PyTorch Tensor the model requires."""
    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # The model expects a batch dimension: [1, 3, 448, 448]
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values.to(DEVICE, dtype=DTYPE)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = str(raw_text).replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(cleaned)
        except Exception:
            return {"raw": cleaned}
            
    if isinstance(result, list):
        return {"detections": result, "skill_name": "Combat Data"}
    elif isinstance(result, dict):
        return result
    return {"raw": cleaned}


def run_chat_variants(image_pil: Image.Image, prompt: str) -> tuple[str, Any]:
    gen_config = {"max_new_tokens": 512, "do_sample": False}
    
    # 1. Convert the image to a Tensor so it has a .shape attribute!
    pixel_values = process_image_to_tensor(image_pil)
    
    # 2. InternVL requires the <image> tag in the prompt
    if "<image>" not in prompt:
        prompt = f"<image>\n{prompt}"

    attempts = [
        ("chat_tensor_positional", lambda: model.chat(tokenizer, pixel_values, prompt, generation_config=gen_config)),
        ("chat_tensor_kw", lambda: model.chat(tokenizer=tokenizer, pixel_values=pixel_values, question=prompt, generation_config=gen_config)),
    ]

    last_error = None
    with torch.inference_mode():
        for method_name, fn in attempts:
            try:
                response = fn()
                # Unpack response if it returns a tuple (response, history)
                if isinstance(response, tuple):
                    response = response[0]
                return method_name, response
            except Exception as exc: 
                last_error = f"{method_name}: {type(exc).__name__}: {exc}"

    raise RuntimeError(f"All chat variants failed. Last error: {last_error}")


class SenseNovaAPI(BaseHTTPRequestHandler):
    def do_POST(self) -> None: 
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            img_path = Path(req.get("image_path", "")).expanduser().resolve()
            prompt = str(req.get("prompt", "")).strip()

            if not img_path.exists():
                self.send_error(404, f"Image not found: {img_path}")
                return

            log(f"--> Received request to analyze: {img_path.name}")
            image = Image.open(img_path).convert("RGB")

            method, response = run_chat_variants(image, prompt)

            result = extract_json_object(str(response))
            result["_bridge_method"] = method
            result["_image_path"] = str(img_path)

            payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            log(f"--> Successfully sent response back to worker!")

        except Exception as exc:  
            error_msg = traceback.format_exc()
            log(f"\n--- SERVER CRASH ---\n{error_msg}\n--------------------\n")
            self.send_error(500, f"Inference failed: {exc}")

    def log_message(self, format: str, *args: object) -> None:  
        return


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), SenseNovaAPI)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Bridge stopped by user.")