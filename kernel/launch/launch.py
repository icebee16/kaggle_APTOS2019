VERSION = "0028"

import subprocess
from pathlib import Path

src_dir = str(Path(__file__).absolute().parents[1] / "input/code_aptos2019/local/src")

proc = subprocess.run(["python", f"{src_dir}/launch_inference.py", f"{VERSION}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(proc.stdout.decode("utf8"))
print(proc.stderr.decode("utf8"))
