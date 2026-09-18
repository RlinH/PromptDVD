#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PromptDVD (formerly VcdPrompt) Cross-Organ: train on one PanNuke organ, eval on the rest."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = os.environ.get(
    "MED_DATASET_ROOT", r"D:\MLDL\PromptDVD\MedDataset"
)

TRAIN_ORGAN = "pannuke_adrenal_gland"
CONFIG_FILE = "configs/trainers/VcdPrompt/vcdprompt-pannuke-cross.yaml"
LOAD_EPOCH = 15
NUM_SHOTS = 32
PANNUKE_EVAL_ORGANS = [
    "pannuke_bile_duct",
    "pannuke_bladder",
    "pannuke_breast",
    "pannuke_cervix",
    "pannuke_colon",
    "pannuke_esophagus",
    "pannuke_headneck",
    "pannuke_kidney",
    "pannuke_liver",
    "pannuke_lung",
    "pannuke_ovarian",
    "pannuke_pancreatic",
    "pannuke_prostate",
    "pannuke_skin",
    "pannuke_stomach",
    "pannuke_testis",
    "pannuke_thyroid",
    "pannuke_uterus",
]


def run_train(args_list):
    cmd = [sys.executable, str(ROOT / "train.py")] + args_list
    return subprocess.run(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    trainer = "VcdPrompt"
    train_output = str(ROOT / "output" / "cross_organ" / "train" / "VcdPrompt")
    test_output = str(ROOT / "output" / "cross_organ" / "eval" / "VcdPrompt")
    opts = ["DATASET.NUM_SHOTS", str(NUM_SHOTS), "DATASET.SUBSAMPLE_CLASSES", "all"]

    print(f"\n========== Cross-Organ 训练: {TRAIN_ORGAN} ==========")
    train_args = [
        "--root", args.root,
        "--seed", str(args.seed),
        "--trainer", trainer,
        "--dataset-config-file", f"configs/datasets/{TRAIN_ORGAN}.yaml",
        "--config-file", CONFIG_FILE,
        "--output-dir", train_output,
    ] + opts
    ret = run_train(train_args)
    if ret.returncode != 0:
        print("训练失败", file=sys.stderr)
        sys.exit(ret.returncode)

    for i, organ in enumerate(PANNUKE_EVAL_ORGANS):
        print(f"\n========== Cross-Organ 评估 [{i + 1}/{len(PANNUKE_EVAL_ORGANS)}] {organ} ==========")
        eval_args = [
            "--root", args.root,
            "--seed", str(args.seed),
            "--trainer", trainer,
            "--dataset-config-file", f"configs/datasets/{organ}.yaml",
            "--config-file", CONFIG_FILE,
            "--output-dir", test_output,
            "--model-dir", train_output,
            "--load-epoch", str(LOAD_EPOCH),
            "--eval-only",
        ] + opts
        ret = run_train(eval_args)
        if ret.returncode != 0:
            print(f"评估失败: {organ}", file=sys.stderr)
            sys.exit(ret.returncode)

    print("\n========== Cross-Organ 全部完成 ==========")


if __name__ == "__main__":
    main()
