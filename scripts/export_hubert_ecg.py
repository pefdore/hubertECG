"""Exporte HuBERT-ECG-SFT-CardioLearning-small (Hugging Face) en ONNX pour hubertECG.

Usage :
    pip install torch transformers safetensors onnx onnxscript
    pip install git+https://github.com/Edoar-do/HuBERT-ECG.git
    python scripts/export_hubert_ecg.py

Produit models/hubert-ecg-small.onnx (entrée `input_values` [batch, 6000] =
12 dérivations x 500 échantillons à 100 Hz, aplaties ; sortie `logits` [batch, 164]).
"""

import os
import sys

import torch
from huggingface_hub import snapshot_download

from hubert_ecg.modeling_classification import HuBERTECGForClassification

REPO = "Edoardo-Coppola/HuBERT-ECG-SFT-CardioLearning-small"
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "hubert-ecg-small.onnx")


def main() -> None:
    local = snapshot_download(REPO)
    model = HuBERTECGForClassification.from_pretrained(local, num_labels=164)
    model.eval()

    dummy = torch.zeros(1, 6000)
    with torch.no_grad():
        logits = model(input_values=dummy, return_dict=True).logits
    assert logits.shape == (1, 164), logits.shape

    torch.onnx.export(
        model,
        (dummy,),
        OUT,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"OK -> {os.path.abspath(OUT)} ({os.path.getsize(OUT) // 1024} Ko)")


if __name__ == "__main__":
    sys.exit(main())
