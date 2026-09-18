#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VcdPrompt Few-Shot experiments on Kather / Colorectal / BloodMNIST / KIMIA."""
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
    base_output = str(ROOT / "output" / "fewshot")
    experiments = [
        {
            "dataset_config": "configs/datasets/kather.yaml",
            "config_file": "configs/trainers/VcdPrompt/vcdprompt-Kather-fs.yaml",
            "output_dir": f"{base_output}/Kather/VcdPrompt",
            "num_shots": 5,
        },
        {
            "dataset_config": "configs/datasets/colorectal.yaml",
            "config_file": "configs/trainers/VcdPrompt/vcdprompt-colorectal-fs.yaml",
            "output_dir": f"{base_output}/colorectal/VcdPrompt",
            "num_shots": 5,
        },
        {
            "dataset_config": "configs/datasets/bloodmnist.yaml",
            "config_file": "configs/trainers/VcdPrompt/vcdprompt-bloodmnist-fs.yaml",
            "output_dir": f"{base_output}/bloodmnist/VcdPrompt",
            "num_shots": 10,
        },
        {
            "dataset_config": "configs/datasets/KIMIA.yaml",
            "config_file": "configs/trainers/VcdPrompt/vcdprompt-KIMIA-fs.yaml",
            "output_dir": f"{base_output}/KIMIA/VcdPrompt",
            "num_shots": 5,
        },
    ]

    for i, exp in enumerate(experiments):
        print(f"\n========== Few-Shot [{i + 1}/{len(experiments)}] {exp['dataset_config']} ==========")
        train_args = [
            "--root", args.root,
            "--seed", str(args.seed),
            "--trainer", trainer,
            "--dataset-config-file", exp["dataset_config"],
            "--config-file", exp["config_file"],
            "--output-dir", exp["output_dir"],
            "DATASET.NUM_SHOTS", str(exp["num_shots"]),
            "DATASET.SUBSAMPLE_CLASSES", "all",
        ]
        ret = run_train(train_args)
        if ret.returncode != 0:
            print(f"实验失败: {exp['dataset_config']}", file=sys.stderr)
            sys.exit(ret.returncode)

    print("\n========== Few-Shot 全部完成 ==========")


if __name__ == "__main__":
    main()
