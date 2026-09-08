import argparse

import numpy as np
import onnxruntime as ort
import torch

from data.dataset import MAX_SEQ_LEN
from models.bilstm import BiLSTMClassifier
from models.transformer import TransformerClassifier

INPUT_DIM = 126


def export(model_name: str, num_classes: int = 100):
    model = (BiLSTMClassifier if model_name == "bilstm" else TransformerClassifier)(num_classes=num_classes)
    model.load_state_dict(torch.load(f"best_{model_name}.pth", map_location="cpu"))
    model.eval()

    dummy_x = torch.randn(1, MAX_SEQ_LEN, INPUT_DIM)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.bool)

    out_path = f"{model_name}.onnx"
    torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        out_path,
        input_names=["landmarks", "mask"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "mask": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported to {out_path}")

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"landmarks": dummy_x.numpy(), "mask": dummy_mask.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy_x, dummy_mask).numpy()
    max_diff = np.abs(onnx_out - torch_out).max()
    print(f"Max diff ONNX vs PyTorch (dummy input): {max_diff:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm", "transformer"], required=True)
    args = parser.parse_args()
    export(args.model)
