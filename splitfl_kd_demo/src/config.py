from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class DataConfig:
    dataset: str = "MNIST"
    data_root: str = "./data"
    image_size: int = 224
    iid: bool = True
    num_clients: int = 6
    selected_clients: int = 3
    num_workers: int = 2

@dataclass
class TrainConfig:
    total_rounds: int = 3
    batch_size: int = 64
    lr_head: float = 1e-3
    lr_tail: float = 1e-3
    lr_student: float = 1e-3
    epochs_head: int = 1
    epochs_tail: int = 1
    epochs_student: int = 1
    server_body_epochs: int = 1
    early_stop_patience: int = 3

@dataclass
class ModelConfig:
    num_classes: int = 10
    split_layer_index: int = 4
    pretrained: bool = True

@dataclass
class KDConfig:
    temperature: float = 4.0
    alpha: float = 0.6
    beta: float = 0.4

@dataclass
class SystemConfig:
    scheme: str = "RANDOM"
    outputs_dir: str = "./outputs"

@dataclass
class AppConfig:
    seed: int = 42
    device: str = "cpu"
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    kd: KDConfig = KDConfig()
    system: SystemConfig = SystemConfig()

def load_config(path: str) -> AppConfig:
    p = Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = AppConfig()
    def upd(dc, d):
        for k, v in d.items():
            if hasattr(dc, k):
                cur = getattr(dc, k)
                if hasattr(cur, "__dataclass_fields__"):
                    upd(cur, v or {})
                else:
                    setattr(dc, k, v)
    upd(cfg, raw)
    return cfg
