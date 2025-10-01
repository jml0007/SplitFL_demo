import argparse, torch
from .config import load_config
from .utils import seed_everything
from .data import make_dataloaders
from .models import build_resnet18_head, build_resnet18_server_body, build_local_tail, build_mobilenetv2_student, ProjectionHead
from .client import Client
from .fl_system import FLSystem

def build_clients(cfg, client_sets, client_loaders, device):
    clients = []
    for cid, dl in enumerate(client_loaders):
        head = build_resnet18_head(cfg.model.split_layer_index, pretrained=cfg.model.pretrained)
        tail = build_local_tail(cfg.model.num_classes)
        student = build_mobilenetv2_student(cfg.model.num_classes, pretrained=cfg.model.pretrained)
        clients.append(Client(client_id=cid, dataset_loader=dl, head=head, tail=tail, student=student, device=device))
    return clients

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--fast_dev_run", action="store_true")
    parsed = parser.parse_args(args=args)

    cfg = load_config(parsed.config)
    if parsed.fast_dev_run:
        cfg.train.total_rounds = 1
        cfg.train.epochs_head = cfg.train.epochs_tail = cfg.train.epochs_student = 1

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    client_sets, client_loaders, public_loader, test_loader = make_dataloaders(
        cfg.data.data_root, cfg.data.image_size, cfg.train.batch_size, cfg.data.num_workers, cfg.data.iid, cfg.data.num_clients
    )

    global_head = build_resnet18_head(cfg.model.split_layer_index, pretrained=cfg.model.pretrained)
    server_body = build_resnet18_server_body(cfg.model.split_layer_index, pretrained=cfg.model.pretrained)
    proj_head   = ProjectionHead(in_dim=512, out_dim=128)

    clients = build_clients(cfg, client_sets, client_loaders, device=device)
    fls = FLSystem(clients=clients, global_head=global_head, server_body=server_body, projection_head=proj_head, device=device)

    fls.pretrain(public_loader=public_loader, epochs_server_body=cfg.train.server_body_epochs, lr=1e-3)

    best_student, wait = 0.0, 0
    for r in range(cfg.train.total_rounds):
        selected = fls.select_clients(cfg.data.selected_clients, scheme=cfg.system.scheme)
        t_acc, s_acc = fls.round_step(
            selected=selected,
            epochs_head=cfg.train.epochs_head,
            epochs_tail=cfg.train.epochs_tail,
            epochs_student=cfg.train.epochs_student,
            lr_head=cfg.train.lr_head,
            lr_tail=cfg.train.lr_tail,
            lr_student=cfg.train.lr_student,
            alpha=cfg.kd.alpha,
            temperature=cfg.kd.temperature
        )
        print(f"[Round {r+1}/{cfg.train.total_rounds}] TeacherAcc={t_acc:.3f} StudentAcc={s_acc:.3f}")
        if s_acc > best_student:
            best_student, wait = s_acc, 0
        else:
            wait += 1
            if wait >= cfg.train.early_stop_patience:
                print(f"Early stopping at round {r+1}. Best Student Acc={best_student:.3f}")
                break

if __name__ == "__main__":
    main()
