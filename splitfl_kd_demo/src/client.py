from dataclasses import dataclass
from typing import Tuple
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from .utils import kd_loss

@dataclass
class Client:
    client_id: int
    dataset_loader: DataLoader
    head: nn.Module
    tail: nn.Module
    student: nn.Module
    device: torch.device

    def local_train_head(self, server_body: nn.Module, epochs: int, lr: float) -> None:
        self.head.to(self.device); self.tail.to(self.device); server_body.to(self.device)
        ce_loss = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.head.parameters(), lr=lr)
        self.head.train(); self.tail.eval(); server_body.eval()
        for _ in range(epochs):
            for x, y in self.dataset_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    h = self.head(x); mid = server_body(h)
                logits = self.tail(mid)
                loss = ce_loss(logits, y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    def local_train_tail(self, server_body: nn.Module, epochs: int, lr: float) -> None:
        self.head.to(self.device); self.tail.to(self.device); server_body.to(self.device)
        ce_loss = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.tail.parameters(), lr=lr)
        self.head.eval(); self.tail.train(); server_body.eval()
        for _ in range(epochs):
            for x, y in self.dataset_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    h = self.head(x); mid = server_body(h)
                logits = self.tail(mid)
                loss = ce_loss(logits, y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    def local_kd_student(self, server_body: nn.Module, epochs: int, lr: float, alpha: float, temperature: float) -> None:
        self.head.to(self.device); self.tail.to(self.device); self.student.to(self.device); server_body.to(self.device)
        opt = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.head.eval(); self.tail.eval(); server_body.eval(); self.student.train()
        for _ in range(epochs):
            for x, y in self.dataset_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    t_logits = self.tail(server_body(self.head(x)))
                s_logits = self.student(x)
                loss = kd_loss(s_logits, t_logits, y, alpha=alpha, temperature=temperature)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    @torch.no_grad()
    def eval_student(self) -> Tuple[float, int]:
        self.student.to(self.device); self.student.eval()
        correct, total = 0, 0
        for x, y in self.dataset_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.student(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        return (correct / max(1, total)), total

    @torch.no_grad()
    def eval_teacher(self, server_body: nn.Module) -> Tuple[float, int]:
        self.head.to(self.device); self.tail.to(self.device); server_body.to(self.device)
        self.head.eval(); self.tail.eval(); server_body.eval()
        correct, total = 0, 0
        for x, y in self.dataset_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.tail(server_body(self.head(x)))
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        return (correct / max(1, total)), total
