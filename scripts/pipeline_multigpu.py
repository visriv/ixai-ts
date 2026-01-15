import argparse
import os
import queue
import multiprocessing as mp
from pathlib import Path

from src.utils.config import load_config, expand_cfg
# from scripts.pipeline import run_single_cfg
from src.utils.plot_samples import plot_sample_timeseries
from scripts.pipeline import run_training, run_pointwise_xai, run_pairwise_xai
from src.utils.loading import load_dataset, save_train_val_pickles

def banner(msg: str):
    print("\n" + "=" * 80)
    print(f"🔹 {msg}")
    print("=" * 80 + "\n")


def run_single_cfg(this_cfg, base_outdir):
    data_gen_flag = bool(this_cfg.get("data_gen", False))
    train_flag = bool(this_cfg.get("train", True))
    pointwise_xai_flag = bool(this_cfg.get("pointwise_xai", True))
    pairwise_xai_flag = bool(this_cfg.get("pairwise_xai", True))

    banner(
        f"model={this_cfg['model'].get('name','?')} "
        f"sweep={this_cfg.get('_sweep_name','?')}"
    )

    if data_gen_flag:
        X, y, A, ds_name = load_dataset(this_cfg["dataset"])
        plot_sample_timeseries(X, ds_name, sample_idx=0)
        (X_train, y_train), (X_val, y_val), data_dir = \
            save_train_val_pickles(X, y, ds_name)
        print(f"📦 Data generated for {ds_name}")

    if train_flag:
        run_training(this_cfg, base_outdir)

    if pointwise_xai_flag:
        run_pointwise_xai(this_cfg, base_outdir)

    if pairwise_xai_flag:
        run_pairwise_xai(this_cfg, base_outdir)
    




def worker(
    gpu_id: int,
    job_queue: mp.Queue,
    base_outdir: str,
):
    # pin this worker to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    print(f"[Worker GPU {gpu_id}] started")

    while True:
        try:
            cfg = job_queue.get_nowait()
        except queue.Empty:
            print(f"[Worker GPU {gpu_id}] done")
            break

        try:
            run_single_cfg(cfg, base_outdir)
        except Exception as e:
            print(f"[Worker GPU {gpu_id}] ERROR: {e}")
            raise e


def main(cfg_path, base_outdir, gpus):
    cfg = load_config(cfg_path)
    all_cfgs = expand_cfg(cfg)

    print(f"Total jobs: {len(all_cfgs)}")
    print(f"Using GPUs: {gpus}")

    job_queue = mp.Queue()
    for c in all_cfgs:
        job_queue.put(c)

    procs = []
    for gpu_id in gpus:
        p = mp.Process(
            target=worker,
            args=(gpu_id, job_queue, base_outdir),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_outdir", default="runs")
    ap.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU ids")
    args = ap.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]

    main(args.config, args.base_outdir, gpu_ids)
