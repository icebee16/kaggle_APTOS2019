import time
import subprocess
import shutil
import tarfile
import json
from pathlib import Path


VERSION = "0029"


def training(version):
    """
    """
    subprocess.run(["python", "src/launch_train.py", version])

    # mkdir for dataset
    (Path(__file__).parent / "kernel" / version).mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(Path(__file__).parent / "kernel" / version / "qwkcoef.tar.gz"), mode="w:gz") as t:
        for coef_path in list((Path(__file__).parent / "model" / "qwkcoef").iterdir()):
            t.add(str(coef_path))

    for f_path in list((Path(__file__).parent / "model").glob(f"{version}*.pth")):
        shutil.copy(str(f_path), str(Path(__file__).parent / "kernel" / version / f_path.name))

    # set metadata
    subprocess.run(["kaggle", "datasets", "init", "-p", f"kernel/{version}"])

    with open(f"kernel/{version}/dataset-metadata.json") as j:
        metadata = json.load(j)

    metadata["title"] = f"model{version}"
    _id_root = metadata["id"].split("/")[0]
    metadata["id"] = f"{_id_root}/model{version}_aptos2019"

    with open(f"kernel/{version}/dataset-metadata.json", mode="w") as j:
        json.dump(metadata, j, indent=2)

    # upload model
    subprocess.run(["kaggle", "datasets", "create", "-p", f"kernel/{version}"])


def update_code(version):
    """
    """
    with tarfile.open(str(Path(__file__).parent / "kernel" / "code" / "local.tar.gz"), mode="w:gz") as t:
        t.add(str(Path(__file__).parent / "config"))
        t.add(str(Path(__file__).parent / "src"))

    # subprocess.run(["kaggle", "datasets", "version", "-p", "kernel/code", "-m", f"update {version}"])
    subprocess.run(["kaggle", "datasets", "create", "-p", "kernel/code"])


def inference(version):
    """
    """
    # set metadata
    with open("kernel/launch/kernel-metadata.json") as j:
        metadata = json.load(j)

    model_source = metadata["dataset_sources"][0]
    metadata["dataset_sources"][0] = model_source.split("/")[0] + f"/model{version}_aptos2019"

    with open("kernel/launch/kernel-metadata.json", mode="w") as j:
        json.dump(metadata, j, indent=2)

    # upload model
    with open((Path(__file__).parent / "kernel" / "launch" / "launch.py"), mode="r+") as f:
        f.write(f"VERSION = \"{version}\"")
    subprocess.run(["kaggle", "kernels", "push", "-p", "kernel/launch"])


def main():
    # training(VERSION)
    update_code(VERSION)
    print("plz wait 3m")
    time.sleep(180)
    inference(VERSION)
    print("done!!!")


if __name__ == "__main__":
    main()
