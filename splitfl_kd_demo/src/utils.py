import os, random, numpy as np, torch

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

def kd_loss(student_logits, teacher_logits, targets, alpha=0.6, temperature=4.0):
    ce = torch.nn.functional.cross_entropy(student_logits, targets)
    p = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
    q = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    kl = torch.nn.functional.kl_div(p, q, reduction="batchmean") * (temperature ** 2)
    return alpha * ce + (1 - alpha) * kl
