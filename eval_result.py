# !/usr/bin/env python
# coding: utf-8

# Max-Heinrich Laves
# Institute of Medical Technology and Intelligent Systems
# Hamburg University of Technology, Germany
# 2021

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set()
from bayesian_optimization import *


def eval(
        task: str,
        bayes: str,
        bo_params: Dict[str, List[float]],
        run_params: Dict
) -> None:
    del run_params['bo_results_path']

    device_list = [torch.device(d) for d in run_params['devices']]
    del run_params['devices']

    candidates = list(itertools.product(*[v["candidates"] for p, v in bo_params.items()]))

    queue = mp.Queue()
    processes = []
    for i, (candidate, dev) in enumerate(zip(candidates, itertools.cycle(device_list))):
        p = mp.Process(target=f, args=(task, bayes, i, queue, candidate, dev, run_params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    y_run = []
    candidates_run = []
    while not queue.empty():
        candidate, res = queue.get()
        candidates_run.append(candidate)
        y_run.append(res)

    # filter nans
    for i, val in enumerate(y_run):
        if np.isnan(val):
            del candidates_run[i]
    y_run = [x for x in y_run if not np.isnan(x)]

    print()
    print(f"{list(bo_params.keys())[0]}      {list(bo_params.keys())[1]}       psnr")
    for c, y in zip(candidates_run, y_run):
        print(f"{c[0]:.6f}  {c[1]:.6f}  {y:.6f}")


if __name__ == '__main__':
    import argparse
    from collections import OrderedDict
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="denoising")
    parser.add_argument("--bayes", type=str, default="mfvi")
    parser.add_argument("--config", type=str, default="./bo_configs/bo_den.json")
    args = parser.parse_args()

    filter_nans = lambda d: {k: v for k, v in d.items() if v is not np.nan}
    try:
        config = pd.read_json(args.config).to_dict(into=OrderedDict)
    except:
        print("Error reading JSON")
        exit()

    bo_params = filter_nans(config["bo_params"])
    run_params = filter_nans(config["run_params"])

    eval(task=args.task,
         bayes=args.bayes,
         bo_params=bo_params,
         run_params=run_params)
