import pandas as pd
import matplotlib.pyplot as plt

import wandb
from wandb import Api
api = Api()          # 若本机已登录 `wandb login`；否则传入 api_key=...
run = api.run("yaojiaqihw3579-1/openpi/opcb90yk")

entity = "yaojiaqihw3579-1"
project = "openpi"
run_id  = "opcb90yk"

run = api.run(f"{entity}/{project}/{run_id}")

# 取完整历史（默认会分页自动拼好；列里包含 step/_step/_timestamp 等）
df = run.history()        # -> pandas.DataFrame
# 也可以只取部分列，加快速度
# df = run.history(keys=["train/loss", "val/acc", "_step"])

# 简单重画
x = df["_step"] if "_step" in df else df["step"]
plt.figure()
plt.plot(x, df["loss"])
plt.xlabel("step")
plt.ylabel("train/loss")
plt.title(f"{run.name} - train/loss")
plt.grid(True, alpha=0.3)
plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
plt.ylim(0, 3)
plt.savefig("./train_loss_plot.png")
