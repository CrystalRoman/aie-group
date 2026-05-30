from pathlib import Path

from src.utils.config import load_yaml


def test_load_yaml_reads_nested_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
paths:
  train_csv_path: dataset_full/stage_1_train_images.csv
training:
  model_name: unet
  learning_rate: 0.0001
""".strip(),
        encoding="utf-8",
    )

    config = load_yaml(str(config_path))

    assert config["seed"] == 42
    assert config["paths"]["train_csv_path"] == "dataset_full/stage_1_train_images.csv"
    assert config["training"]["model_name"] == "unet"
