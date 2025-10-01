from typing import List, Tuple
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from .client import Client
from .models import ProjectionHead

def fedavg(models: List[nn.Module]) -> nn.Module:
    base = models[0]
    sd_list = [m.state_dict() for m in models]
    avg_sd = {}
    for k in sd_list[0].keys():
        avg_sd[k] = sum(sd[k] for sd in sd_list) / len(sd_list)
    base.load_state_dict(avg_sd)
    return base

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t())
    n = z1.size(0)
    mask = torch.eye(2*n, device=z.device, dtype=torch.bool)
    sim = sim / temperature
    targets = torch.cat([torch.arange(n, 2*n), torch.arange(0, n)]).to(z.device)
    sim = sim.masked_fill(mask, float('-inf'))
    log_prob = torch.nn.functional.log_softmax(sim, dim=1)
    positives = log_prob[torch.arange(2*n), targets]
    return -positives.mean()

@dataclass
class FLSystem:
    clients: List[Client]
    global_head: nn.Module
    server_body: nn.Module
    projection_head: ProjectionHead
    device: torch.device

    def pretrain(self, public_loader: DataLoader, epochs_server_body: int, lr: float = 1e-3):
        """SimCLR on server_body+projection_head using public data; global_head is a frozen encoder."""
        self.server_body.to(self.device); self.projection_head.to(self.device); self.global_head.to(self.device)
        opt = torch.optim.Adam(list(self.server_body.parameters()) + list(self.projection_head.parameters()), lr=lr)
        self.server_body.train(); self.projection_head.train(); self.global_head.eval()
        for _ in range(epochs_server_body):
            for x1, x2 in public_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                with torch.no_grad():
                    h1 = self.global_head(x1); h2 = self.global_head(x2)
                z1 = self.projection_head(self.server_body(h1))
                z2 = self.projection_head(self.server_body(h2))
                loss = nt_xent(z1, z2, temperature=0.5)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    def select_clients(self, k: int, scheme: str = "RANDOM") -> List[Client]:
        import random
        if scheme.upper() == "RANDOM":
            return random.sample(self.clients, k)
        # 'PRIORITY': pick clients whose student accuracy is currently lowest
        accs = []
        for c in self.clients:
            acc, _ = c.eval_student()
            accs.append((acc, c))
        accs.sort(key=lambda x: x[0])
        return [c for _, c in accs[:k]]

    def round_step(self,
                   selected: List[Client],
                   epochs_head: int,
                   epochs_tail: int,
                   epochs_student: int,
                   lr_head: float,
                   lr_tail: float,
                   lr_student: float,
                   alpha: float,
                   temperature: float) -> Tuple[float, float]:
        for c in selected:
            c.local_train_head(self.server_body, epochs=epochs_head, lr=lr_head)
        for c in selected:
            c.local_train_tail(self.server_body, epochs=epochs_tail, lr=lr_tail)
        for c in selected:
            c.local_kd_student(self.server_body, epochs=epochs_student, lr=lr_student, alpha=alpha, temperature=temperature)
        avg_head = fedavg([c.head for c in selected])
        self.global_head.load_state_dict(avg_head.state_dict())
        with torch.no_grad():
            t_accs, s_accs = [], []
            for c in selected:
                t_acc, _ = c.eval_teacher(self.server_body)
                s_acc, _ = c.eval_student()
                t_accs.append(t_acc); s_accs.append(s_acc)
            return float(sum(t_accs)/len(t_accs)), float(sum(s_accs)/len(s_accs))
