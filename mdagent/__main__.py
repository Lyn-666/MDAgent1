import os
import sys

try:
    from .cli import main
except ModuleNotFoundError as e:
    # Usually the wrong conda environment (for example "base"): say how to fix it instead of a traceback.
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    where = f"the conda environment '{env}'" if env else "this Python"
    where_zh = f"conda 环境 {env}" if env else "这个 Python"
    sys.exit(f"错误：{where_zh} 里没有装 {e.name}。请先运行 conda activate mdagent，再运行这条命令。\n"
             f"error: {e.name} is not installed in {where}. Activate the MDAgent environment first:\n"
             f"    conda activate mdagent\n"
             f"then run the command again.")

sys.exit(main())
