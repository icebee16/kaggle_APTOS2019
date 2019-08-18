import subprocess
import shutil
import tarfile
import json
from pathlib import Path


VERSION_LIST = ["0045", "0046"]


def training(version):
    """
    """
    subprocess.run(["python", "src/launch_train.py", version])

    dataset_dir = Path(__file__).parent / "kernel" / "input" / f"model{version}_aptos2019"

    # mkdir for dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(dataset_dir / "qwkcoef.tar.gz"), mode="w:gz") as t:
        for coef_path in list((Path(__file__).parent / "model" / "qwkcoef").iterdir()):
            t.add(str(coef_path))

    for f_path in list((Path(__file__).parent / "model").glob(f"{version}*.pth")):
        shutil.copy(str(f_path), str(dataset_dir / f_path.name))

    # set metadata
    subprocess.run(["kaggle", "datasets", "init", "-p", f"kernel/input/model{version}_aptos2019"])

    with open(dataset_dir / "dataset-metadata.json") as j:
        metadata = json.load(j)

    metadata["title"] = f"model{version}_aptos2019"
    _id_root = metadata["id"].split("/")[0]
    metadata["id"] = f"{_id_root}/model{version}_aptos2019"

    with open(dataset_dir / "dataset-metadata.json", mode="w") as j:
        json.dump(metadata, j, indent=2)

    # upload model
    subprocess.run(["kaggle", "datasets", "create", "-p", f"kernel/input/model{version}_aptos2019"])


#  def update_code(version):
#      """
#      """
#      with tarfile.open(str(Path(__file__).parent / "kernel" / "code" / "local.tar.gz"), mode="w:gz") as t:
#          t.add(str(Path(__file__).parent / "config"))
#          t.add(str(Path(__file__).parent / "src"))
#
#      # subprocess.run(["kaggle", "datasets", "version", "-p", "kernel/code", "-m", f"update {version}"])
#      subprocess.run(["kaggle", "datasets", "create", "-p", "kernel/code"])
#
#
#  def inference(version):
#      """
#      """
#      # set metadata
#      with open("kernel/launch/kernel-metadata.json") as j:
#          metadata = json.load(j)
#
#      model_source = metadata["dataset_sources"][0]
#      metadata["dataset_sources"][0] = model_source.split("/")[0] + f"/model{version}_aptos2019"
#
#      with open("kernel/launch/kernel-metadata.json", mode="w") as j:
#          json.dump(metadata, j, indent=2)
#
#      # upload model
#      with open((Path(__file__).parent / "kernel" / "launch" / "launch.py"), mode="r+") as f:
#          f.write(f"VERSION = \"{version}\"")
#      subprocess.run(["kaggle", "kernels", "push", "-p", "kernel/launch"])


def inference(version):
    """
    """
    notebook_dir = Path(__file__).parent / "kernel" / version
    # notebook_title = f"{version}_aptos2019"

    # mkdir for dataset
    notebook_dir.mkdir(parents=True, exist_ok=True)

    # set metadata
    subprocess.run(["kaggle", "kernels", "init", "-p", f"kernel/{version}"])

    with open(notebook_dir / "kernel-metadata.json") as j:
        metadata = json.load(j)

    username = metadata["id"].split("/")[0]
    metadata["id"] = username + "/" + version + "-aptos2019"
    metadata["title"] = f"{version} aptos2019"
    metadata["code_file"] = "inference.py"
    metadata["language"] = "python"
    metadata["kernel_type"] = "script"
    metadata["is_private"] = "true"
    metadata["enable_gpu"] = "true"
    metadata["enable_internet"] = "false"
    metadata["dataset_sources"] = [username + "/" + f"model{version}_aptos2019"]
    metadata["competition_sources"] = ["aptos2019-blindness-detection"]
    metadata["kernel_sources"] = []

    with open(notebook_dir / "kernel-metadata.json", mode="w") as j:
        json.dump(metadata, j, indent=2)


def main():
    for v in VERSION_LIST:
        training(v)
        inference(v)


if __name__ == "__main__":
    main()
