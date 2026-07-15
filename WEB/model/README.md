# PLACEHOLDER — drop the trained model here

The web app looks for **`surrogate_unet.onnx`** in this folder. Until it
exists, [index.html](../index.html) runs in *FSPL preview mode* (free-space
path loss only, walls ignored) and shows a banner.

To produce it:

1. Run `STEP_4/train_surrogate_colab.ipynb` on Colab (GPU runtime).
2. Run the last cell ("Export for the interactive web app") — it converts the
   best checkpoint to ONNX and downloads `surrogate_unet.onnx`.
3. Put the file here: `WEB/model/surrogate_unet.onnx`.
4. Serve the app: `python3 -m http.server --directory WEB` and open
   `http://localhost:8000`.

The model expects a 1×3×256×568 float32 input (wall dB / 20, clutter dB-per-m
/ 0.3, log10(distance m) / 3) and outputs path loss / 150 — the constants live
in `WEB/assets/meta.json` and are already wired into the app.
