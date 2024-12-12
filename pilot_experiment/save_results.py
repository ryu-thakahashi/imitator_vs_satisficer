from pathlib import Path
from icecream import ic
import os
import re
import matplotlib.pyplot as plt
import pymc as pm


dir_path = Path(__file__).resolve().parents[0]
res_path = dir_path / "results"
fig_path = dir_path / "figures"


def mkdir_if_not_exist(path: Path):
    if not path.exists():
        path.mkdir(parents=True)


def save_fig(fig_name: str):
    mkdir_if_not_exist(fig_path)
    fig_file_name = re.sub(r"\.png$", "", fig_name)
    plt.savefig(fig_path / (fig_file_name + ".png"))
    plt.close()


def save_res(res_name: str, summary):
    mkdir_if_not_exist(res_path)
    res_file_name = re.sub(r"\.txt$", "", res_name) + ".txt"
    with open(res_path / res_file_name, "w") as f:
        f.write(summary.to_string(max_rows=None, max_cols=None))
    print(f"Saved {res_name} to {res_path / res_file_name}")


def save_graphviz(model, model_name):
    mkdir_if_not_exist(fig_path)
    g = pm.model_to_graphviz(model)

    g.render(fig_path / model_name, format="png", cleanup=True)
    print(f"Saved {model_name} to {fig_path / (model_name + '.png')}")
