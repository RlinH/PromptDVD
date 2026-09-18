#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PromptDVD (formerly VcdPrompt) Base-to-New on Kather / Colorectal / BloodMNIST / KIMIA."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = os.environ.get(
    "MED_DATASET_ROOT", r"D:\MLDL\PromptDVD\MedDataset"
)


def run_train(args_list):
    cmd = [sys.executable, str(ROOT / "train.py")] + args_list
    return subprocess.run(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    trainer = "VcdPrompt"
    base_out = str(ROOT / "output" / "base2new" / "train_base")
    test_out = str(ROOT / "output" / "base2new" / "test_new")
    experiments = [
        {
            "name": "Kather",
            "dataset_config": "configs/datasets/kather.yaml",
            "config_file": "configs/trainers/VcdPrompt/vcdprompt-Kather.yaml",
            "train_output": f"{base_out}/Kather/VcdPrompt",
            "test_output": f"{test_out}/Kather/VcdPrompt",
            "num_shots": 32,
            "load_epoch": 15,
        },
        {
            "name": "Colorectal",
            "dataset_config": "configs/datasets/colorectal.yaml",
            "config_file": "configs/trainers/VcdPrompt/vcdprompt-colorectal.yaml",
            "train_output": f"{base_out}/colorectal/VcdPrompt",
            "test_output": f"{test_out}/colorectal/VcdPrompt",
            "num_shots": 32,
            "load_epoch": 15,
        },
        {
            "name": "Bloodmnist",
            "dataset_config": "configs/datasets/bloodmnist.yaml",
            "config_file": "configs/trainers/VcdPrompt/vcdprompt-bloodmnist.yaml",
            "train_output": f"{base_out}/bloodmnist/VcdPrompt",
            "test_output": f"{test_out}/bloodmnist/VcdPrompt",
            "num_shots": 32,
            "load_epoch": 20,
        },
        {
            "name": "KIMIA",
            "dataset_config": "configs/datasets/KIMIA.yaml",
            "config_file": "configs/trainers/VcdPrompt/vcdprompt-KIMIA.yaml",
            "train_output": f"{base_out}/KIMIA/VcdPrompt",
            "test_output": f"{test_out}/KIMIA/VcdPrompt",
            "num_shots": 32,
            "load_epoch": 15,
        },
    ]

    for i, exp in enumerate(experiments):
        print(f"\n========== Generalization [{i + 1}/{len(experiments)}] {exp['name']} ==========")
        train_args = [
            "--root", args.root,
            "--seed", str(args.seed),
            "--trainer", trainer,
            "--dataset-config-file", exp["dataset_config"],
            "--config-file", exp["config_file"],
            "--output-dir", exp["train_output"],
            "DATASET.NUM_SHOTS", str(exp["num_shots"]),
            "DATASET.SUBSAMPLE_CLASSES", "base",
        ]
        ret = run_train(train_args)
        if ret.returncode != 0:
            print(f"训练失败: {exp['name']}", file=sys.stderr)
            sys.exit(ret.returncode)

        eval_args = [
            "--root", args.root,
            "--seed", str(args.seed),
            "--trainer", trainer,
            "--dataset-config-file", exp["dataset_config"],
            "--config-file", exp["config_file"],
            "--output-dir", exp["test_output"],
            "--model-dir", exp["train_output"],
            "--load-epoch", str(exp["load_epoch"]),
            "--eval-only",
            "DATASET.NUM_SHOTS", str(exp["num_shots"]),
            "DATASET.SUBSAMPLE_CLASSES", "all",
        ]
        ret = run_train(eval_args)
        if ret.returncode != 0:
            print(f"评估失败: {exp['name']}", file=sys.stderr)
            sys.exit(ret.returncode)

    print("\n========== Generalization 全部完成 ==========")


if __name__ == "__main__":
    main()
