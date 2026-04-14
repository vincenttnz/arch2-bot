import json
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "sensenova/SenseNova-SI-1.1-InternVL3-2B"
MAX_FRAMES = 5   # change this after it works


def import_torch_or_explain():
    try:
        import torch
        return torch
    except OSError as e:
        raise RuntimeError(
            "PyTorch could not load. Check the DLL/runtime fix first."
        ) from e


def get_device_and_dtype(torch):
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


def run_chat_variants(model, tokenizer, image, prompt, torch):
    generation_config = {"max_new_tokens": 512, "do_sample": False}
    attempts = [
        ("chat_positional", lambda: model.chat(tokenizer, image, prompt, generation_config=generation_config)),
        ("chat_keyword_image", lambda: model.chat(tokenizer=tokenizer, image=image, question=prompt, generation_config=generation_config)),
        ("chat_keyword_question", lambda: model.chat(tokenizer=tokenizer, image=image, prompt=prompt, generation_config=generation_config)),
    ]

    last_error = None
    for name, fn in attempts:
        try:
            with torch.inference_mode():
                out = fn()
            return name, out
        except Exception as e:
            last_error = f"{name}: {type(e).__name__}: {e}"

    raise RuntimeError(last_error or "All chat variants failed.")


def build_dataset():
    torch = import_torch_or_explain()

    raw_frames_dir = Path("../data/raw_frames")
    output_file = Path("../data/combat_boxes.jsonl")

    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(raw_frames_dir.glob("*.png"))

    if not frames:
        print(f"No frames found in {raw_frames_dir.resolve()}!")
        return

    print("Loading SenseNova model via Transformers...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

    device, dtype = get_device_and_dtype(torch)
    print(f"Using device: {device}, dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().to(device)

    prompt = """Analyze this mobile game screen.
1. Classify the screen type (combat, skill_selection, pause, menu).
2. Provide the [x1, y1, x2, y2] bounding box for the player character.
3. Provide the bounding boxes for all enemies.
4. Provide the bounding boxes for all projectiles.
Return ONLY valid JSON."""

    dataset = []
    test_frames = frames[:MAX_FRAMES]
    print(f"Found {len(frames)} frames. Debug mode will process only {len(test_frames)} frame(s).")

    for img_path in test_frames:
        print(f"Processing {img_path.name}...")
        image = Image.open(img_path).convert("RGB")

        try:
            method, response = run_chat_variants(model, tokenizer, image, prompt, torch)
            print(f"Worked with: {method}")
        except Exception as e:
            print(f"Failed on {img_path.name}: {e}")
            continue

        if isinstance(response, tuple):
            response = response[0]

        cleaned = str(response).replace("```json", "").replace("```", "").strip()

        try:
            labels = json.loads(cleaned)
            labels["frame_id"] = img_path.name
            dataset.append(labels)
            print(f"Parsed JSON for {img_path.name}")
        except json.JSONDecodeError:
            print(f"Non-JSON response for {img_path.name}:\n{response}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Saved {len(dataset)} item(s) to: {output_file.resolve()}")


if __name__ == "__main__":
    build_dataset()
