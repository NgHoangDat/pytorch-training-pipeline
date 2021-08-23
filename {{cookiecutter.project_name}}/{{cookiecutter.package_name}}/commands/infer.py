import json
import os
import sys
from pathlib import Path
from typing import *

import torch
import torch.onnx
from torch.nn import Module
from typer import Argument, Option

from .app import app

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from core import *
from training import *


@app.command()
def infer(
    model_dir: str = Argument(..., help="Model directory"),
    test_files: List[str] = Argument(..., help="Path to test file or directory"),
    output_path: str = Option("", help="Path to output file or directory"),
    gpu: int = Option(-1, help="Gpu to use"),
):
    print(f"Infer {len(test_files)} files")

    model: Model = load_from_dir(model_dir)
    model.eval()

    device = torch.device("cpu")
    if gpu > -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")

    model.to(device)

    predictions = []  # model prediction put here
    if output_path:
        if output_path.endswith(".json"):
            for fn, prediction in zip(test_files, predictions):
                prediction["path"] = fn

            output_path: Path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=4, sort_keys=True, ensure_ascii=False)
        else:
            Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
            for fn, prediction in zip(test_files, predictions):
                fn = Path(fn).resolve()

                with open(
                    os.path.join(output_path, f"{fn.stem}.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        prediction, f, indent=4, sort_keys=True, ensure_ascii=False
                    )
    else:
        for fn, prediction in zip(test_files, predictions):
            print(f"\n{fn}")
            for key, val in prediction.items():
                print(f"{key} {val}")
