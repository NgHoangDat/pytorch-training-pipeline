import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from shutil import copyfile, rmtree
from typing import *

import pandas as pd
import torch

from torch.utils.data import DataLoader
from typer import Argument, Option, progressbar

from .app import app

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from core import *
from training import *


METRICS = {}

KEY_METRIC = "f1"
MAX_SUPPORTED_DISP_LABELS = 100


@app.command()
def test(
    checkpoint_dir: str = Argument(..., help="Checkpoint directory"),
    test_files: List[str] = Argument(..., help="Test csv file"),
    variant: str = Option("best_val_loss", help="Checkpoint variant"),
    latest: bool = Option(True, help="Only test latest version"),
    key_metric: str = Option(KEY_METRIC, help="Key metric to show"),
    output_dir: str = Option("", help="Output dir"),
):

    if len(test_files) <= 1:
        print("Run test on:", test_files[0])
    else:
        print("Run test on:")
        for fn in test_files:
            print("-", fn)

    directory = Path(checkpoint_dir).resolve()

    target_dir = ""
    if output_dir:
        target_dir = os.path.join(output_dir, directory.stem)
        print("Result:", target_dir)

    test_model(
        directory,
        test_files=test_files,
        variant=variant,
        latest=latest,
        key_metric=key_metric,
        output_dir=target_dir,
    )


def test_model(
    checkpoint_dir: Path,
    test_files: List[str] = None,
    variant: str = "best_val_loss",
    latest: bool = True,
    key_metric: str = KEY_METRIC,
    output_dir: str = "",
):
    config_path = checkpoint_dir.joinpath("config.json").as_posix()

    config = TrainingConfig.parse_file(config_path)
    model_cls: Type[Model] = model_registry.get_class(config.model.type)

    if model_cls is None:
        raise ValueError(f"Model {config.model.type} is not supported")

    model_cfg: ModelConfig = model_cls.get_config_cls()(**config.model.params)
    model: Model = model_cls(model_cfg)

    test_dataset = dataset_registry.get(config.val_dataset.type)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    latest_version = ""
    if latest:
        variants = [fn.stem for fn in checkpoint_dir.rglob("best_val_loss*.ckpt")]
        variants.sort()
        match = re.search("(?<=best_val_loss).+", variants[-1])
        if match:
            latest_version = match.group()

    if variant == "all":
        checkpoints_path = list(checkpoint_dir.rglob(f"*{latest_version}.ckpt"))
    else:
        checkpoints_path = [checkpoint_dir.joinpath(f"{variant}{latest_version}.ckpt")]

    outputs = defaultdict(list)

    if torch.cuda.is_available():
        n_gpus = config.n_gpus
        cuda_idx = torch.cuda.current_device()

        if type(n_gpus) is list:
            cuda_idx = config.n_gpus[0]

        if type(n_gpus) is str:
            cuda_idx = int(n_gpus)

        device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = torch.device("cpu")

    if output_dir:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir.joinpath("config.json"), "w") as f:
            json.dump(config.dict(), f, indent=4, sort_keys=True)

    for checkpoint_path in checkpoints_path:
        output, golds, preds = test_variant(
            model,
            device,
            config=config,
            checkpoint_path=checkpoint_path,
            test_dataloader=test_dataloader,
            key_metric=key_metric,
        )
        outputs["variant"].append(Path(checkpoint_path).stem)

        for key, val in output.items():
            outputs[key].append(val)

        if output_dir:
            variant_dir = output_dir.joinpath(checkpoint_path.stem)

            if variant_dir.exists():
                rmtree(variant_dir.as_posix())

            variant_dir.mkdir(parents=True, exist_ok=True)

            cate_dir = variant_dir.joinpath(key)
            cate_dir.mkdir(parents=True, exist_ok=True)

            for fn, gold, pred in zip(test_files, golds, preds):
                if gold != pred:
                    fn = Path(fn)
                    copyfile(
                        fn.as_posix(),
                        cate_dir.joinpath(
                            f"{gold}-{pred}-{fn.parent.stem}-{fn.name}"
                        ).as_posix(),
                    )

    if output_dir:
        output_file = output_dir.joinpath(f"{checkpoint_dir.stem}.csv").as_posix()
        pd.DataFrame(data=outputs).to_csv(output_file, index=False, encoding="utf-8")


def test_variant(
    model: Model,
    device: str,
    config: TrainingConfig,
    checkpoint_path: Path,
    test_dataloader: DataLoader,
    key_metric: str = KEY_METRIC,
) -> Dict[str, float]:
    print("Test for", checkpoint_path.stem)

    wrapper = wrapper_registry.get_class(config.wrapper.type).load_from_checkpoint(
        checkpoint_path=checkpoint_path.as_posix(),
        model=model,
        strict=False,
        **config.wrapper.params,
    )
    wrapper.eval()
    wrapper.to(device)

    preds = []
    golds = []

    with progressbar(test_dataloader) as batches:
        for batch in batches:
            inputs, gold = batch
            inputs = inputs.to(device)

            pred = wrapper.forward(inputs)

            golds.extend(gold.detach().cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())

    df = {"metric": [], "score": []}

    output = {}

    for metric, func in METRICS.items():
        score = func(golds, preds)

        df["metric"].append(metric)
        df["score"].append(score)

        output[f"{metric}"] = score

    df = pd.DataFrame(data=df)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("\nResult:")
        print(df.loc[df["metric"] == key_metric].to_string(index=False))
        print("=" * 80)

    return output, golds, preds
