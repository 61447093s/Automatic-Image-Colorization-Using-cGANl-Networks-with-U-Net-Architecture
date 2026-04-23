import os
import sys
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from skimage.color import rgb2lab, lab2rgb
import torch

sys.path.append(os.path.expanduser("~/colorizer_gan/notebooks"))
from model import Generator

# ---- Config ----
MODEL_PATH  = os.path.expanduser("~/colorizer_gan/models/checkpoint_epoch_100.pth")
UPLOAD_PATH = os.path.expanduser("~/colorizer_gan/app/uploads")
IMAGE_SIZE  = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(UPLOAD_PATH, exist_ok=True)

app = Flask(__name__)

# ---- Load Model ----
def load_generator(model_path, device):
    G          = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    G.load_state_dict(checkpoint["G_state"])
    G.eval()
    print(f"✅ Generator loaded — epoch {checkpoint['epoch']}")
    return G

G = load_generator(MODEL_PATH, DEVICE)
print(f"✅ Device: {DEVICE}")

# ---- Helper Functions ----
def detect_image_type(img_np):
    r       = img_np[:, :, 0].astype("float32")
    g       = img_np[:, :, 1].astype("float32")
    b       = img_np[:, :, 2].astype("float32")
    avg_diff = (np.mean(np.abs(r - g)) +
                np.mean(np.abs(r - b)) +
                np.mean(np.abs(g - b))) / 3.0
    return "grayscale" if avg_diff < 2.0 else "color"


def colorize(img_np, G, device, image_size=256):
    img_np = np.array(Image.fromarray(img_np).resize(
        (image_size, image_size), Image.BICUBIC)) / 255.0

    lab    = rgb2lab(img_np).astype("float32")
    L      = (lab[:, :, 0] / 50.0) - 1.0
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        AB_pred = G(L_tensor).cpu().squeeze(0)

    L_np             = (L_tensor.cpu().squeeze().numpy() + 1.0) * 50.0
    AB_np            = AB_pred.permute(1, 2, 0).numpy() * 128.0

    lab_out          = np.zeros((image_size, image_size, 3), dtype="float32")
    lab_out[:, :, 0] = L_np
    lab_out[:, :, 1:]= AB_np

    rgb_colorized    = np.clip(lab2rgb(lab_out), 0, 1)
    gray             = np.stack([L_np / 100.0] * 3, axis=-1)

    return gray, rgb_colorized


def to_grayscale(img_np, image_size=256):
    img_np = np.array(Image.fromarray(img_np).resize(
        (image_size, image_size), Image.BICUBIC)) / 255.0
    lab    = rgb2lab(img_np).astype("float32")
    L_np   = lab[:, :, 0] / 100.0
    gray   = np.clip(np.stack([L_np] * 3, axis=-1), 0, 1)
    return img_np, gray


def numpy_to_base64(img_np):
    img_np  = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    buffer  = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read image
    img     = Image.open(file.stream).convert("RGB")
    img_np  = np.array(img)

    # Get mode from request
    mode    = request.form.get("mode", "auto")

    # Autodetect if needed
    if mode == "auto":
        detected = detect_image_type(img_np)
    else:
        detected = mode

    # Process based on detected type
    if detected == "grayscale":
        input_img, output_img = colorize(img_np, G, DEVICE, IMAGE_SIZE)
        mode_label = "Grayscale → Color"
    else:
        input_img, output_img = to_grayscale(img_np, IMAGE_SIZE)
        mode_label = "Color → Grayscale"

    # Convert to base64 for sending to frontend
    input_b64  = numpy_to_base64(input_img)
    output_b64 = numpy_to_base64(output_img)

    # Save output file for download
    output_pil  = Image.fromarray((output_img * 255).astype(np.uint8))
    save_path   = os.path.join(UPLOAD_PATH, "latest_result.png")
    output_pil.save(save_path)

    return jsonify({
        "input":      input_b64,
        "output":     output_b64,
        "mode":       mode_label,
        "detected":   detected
    })


@app.route("/download")
def download():
    save_path = os.path.join(UPLOAD_PATH, "latest_result.png")
    if not os.path.exists(save_path):
        return jsonify({"error": "No result to download"}), 404
    return send_file(save_path, as_attachment=True, download_name="colorized_result.png")


# ---- Run ----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)