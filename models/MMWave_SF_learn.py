# ============================================================
# File: doppler_upsample.py
# Author: Jolin (改写)
# Date:
# Description:
#   基于 Doppler 的点云上采样（三支路）
#   - 支持将 quantile 设为可学习参数（默认可学习）
#   - 使用软秩近似计算可微阈值，使 quantile 能被 loss 优化
# Revision 260114：返回fast out
# ============================================================

from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DopplerUpsampler(nn.Module):
    """
    Doppler-based upsampler.
    - 输入 x : Tensor [B, T, P, C]
    - 输出:
        out      : [B, T, max_point, C]
        fast_out : [B, T, fast_max_point, C]
        info     : dict（统计信息）
    """

    def __init__(
        self,
        doppler_idx: int = 3,
        init_quantile: float = 0.3,
        learnable: bool = True,
        fast_scale: int = 2,
        max_point: int = 1024,
        temperature: float = 1.0,
        rank_sigma_frac: float = 0.05,
        shuffle: bool = True,
    ):
        super().__init__()

        assert 0.0 < init_quantile < 1.0
        assert fast_scale >= 1
        assert max_point >= 1

        self.doppler_idx = int(doppler_idx)
        self.fast_scale = int(fast_scale)
        self.max_point = int(max_point)
        self.temperature = float(temperature)
        self.rank_sigma_frac = float(rank_sigma_frac)
        self.shuffle = bool(shuffle)

        logit_q = float(np.log(init_quantile / (1.0 - init_quantile)))
        self.logit_q = nn.Parameter(
            torch.tensor(logit_q, dtype=torch.float32),
            requires_grad=bool(learnable),
        )
        if not learnable:
            self.logit_q.requires_grad_(False)

    def quantile(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_q)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        B, T, P, C = x.shape
        device, dtype = x.device, x.dtype

        out = torch.zeros((B, T, self.max_point, C), device=device, dtype=dtype)

        info = {
            "fast_counts": [],
            "slow_counts": [],
            "after_dup_counts": [],
        }

        # ===== FAST_OUT: 先用 list 收集 =====
        fast_list = [[None for _ in range(T)] for _ in range(B)]

        q = self.quantile().to(device=device, dtype=dtype)
        sigma_base = max(1.0, float(P) * self.rank_sigma_frac)

        for b in range(B):
            for t in range(T):
                frame = x[b, t]                     # [P, C]
                doppler = frame[:, self.doppler_idx].abs()  # [P]

                if P == 0 or doppler.sum() == 0:
                    info["fast_counts"].append(0)
                    info["slow_counts"].append(0)
                    info["after_dup_counts"].append(0)
                    fast_list[b][t] = None
                    continue

                # ---------- 1) soft quantile threshold ----------
                sorted_vals, _ = torch.sort(doppler)
                ranks = torch.arange(P, device=device, dtype=dtype)
                target_idx = q * (P - 1)

                diff = ranks - target_idx
                weights = torch.softmax(
                    -(diff ** 2) / (2.0 * sigma_base ** 2), dim=0
                )
                thresh = (weights * sorted_vals).sum()

                # ---------- 2) fast probability ----------
                logits = (doppler - thresh) / (self.temperature + 1e-12)
                fast_prob = torch.sigmoid(logits)

                # ---------- 3) STE mask ----------
                hard_mask = (doppler > thresh)
                fast_mask = (hard_mask.float() - fast_prob).detach() + fast_prob

                fast_pts = frame * fast_mask.unsqueeze(1)
                slow_pts = frame * (1.0 - fast_mask).unsqueeze(1)

                with torch.no_grad():
                    Pf = int(hard_mask.sum().item())
                    Ps = P - Pf
                    info["fast_counts"].append(Pf)
                    info["slow_counts"].append(Ps)

                # ===== FAST_OUT: 保存硬 fast 点（排序后）=====
                with torch.no_grad():
                    if Pf > 0:
                        fast_raw = frame[hard_mask]          # [Pf, C]
                        d = doppler[hard_mask]
                        order = torch.argsort(d, descending=True)
                        fast_list[b][t] = fast_raw[order]    # 按 doppler 排序
                    else:
                        fast_list[b][t] = None

                # ---------- 4) fast duplication ----------
                if Pf > 0:
                    fast_dup = fast_pts.repeat_interleave(self.fast_scale, dim=0)
                else:
                    top_idx = torch.argmax(doppler)
                    fast_dup = frame[top_idx:top_idx + 1].repeat(
                        self.fast_scale, 1
                    )

                merged = torch.cat([fast_dup, slow_pts], dim=0)
                M = merged.shape[0]
                info["after_dup_counts"].append(int(M))

                # ---------- 5) pad / sample ----------
                if M >= self.max_point:
                    idx = torch.randperm(M, device=device)[: self.max_point]
                    final = merged[idx]
                else:
                    need = self.max_point - M
                    w = fast_prob.detach() + 1e-12
                    w = w / w.sum()
                    idx = torch.multinomial(w, need, replacement=True)
                    final = torch.cat([merged, frame[idx]], dim=0)

                if self.shuffle:
                    final = final[torch.randperm(final.shape[0], device=device)]

                out[b, t] = final

        # ==========================================================
        # FAST_OUT: list -> tensor（统一 fast_max_point）
        # ==========================================================
        fast_max_point = max(info["fast_counts"]) if len(info["fast_counts"]) > 0 else 0
        fast_out = torch.zeros(
            (B, T, fast_max_point, C), device=device, dtype=dtype
        )

        with torch.no_grad():
            for b in range(B):
                for t in range(T):
                    fast_bt = fast_list[b][t]

                    if fast_bt is None or fast_bt.shape[0] == 0:
                        # 退化：复制 doppler 最大的 raw 点
                        doppler = x[b, t, :, self.doppler_idx].abs()
                        idx = torch.argmax(doppler)
                        fast_bt = x[b, t, idx:idx + 1]

                    if fast_bt.shape[0] >= fast_max_point:
                        fast_out[b, t] = fast_bt[:fast_max_point]
                    else:
                        need = fast_max_point - fast_bt.shape[0]
                        repeat = fast_bt[:1].repeat(need, 1)
                        # repeat = fast_bt[:need]
                        fast_out[b, t] = torch.cat([fast_bt, repeat], dim=0)

        return out, fast_out, info

# ============================================================
# main: 简单测试与梯度检查示例
# ============================================================
if __name__ == "__main__":
    # 简单单元测试：确认 quantile 能获取梯度并被优化器更新
    torch.manual_seed(0)

    B, T, P, C = 2, 32, 64, 5
    max_point = 1024

    # 模拟 [x,y,z,doppler,intensity]
    x = torch.randn(B, T, P, C)
    x[..., 3] *= 3.0  # 放大 Doppler

    # 可学习 quantile 的上采样器
    ups = DopplerUpsampler(
        doppler_idx=3,
        init_quantile=0.3,  # 0.3   0.01
        learnable=True,
        fast_scale=2,
        max_point=max_point,
        temperature=0.5,
        rank_sigma_frac=0.06,
        shuffle=True,
    )

    # 简单把参数加入 optimizer（通常你会把 upsampler 嵌入到你的 model 中，
    # 那样 model.parameters() 会包含 quantile）
    opt = torch.optim.SGD(ups.parameters(), lr=0.1)

    # forward
    out, fast_out, info = ups(x)  # out shape [B, T, max_point, C]

    # 构造简单 loss（例如把输出来自某通道的均值作为损失）
    loss = out[..., 0].mean()  # 简单的示例 loss

    # backward
    loss.backward()

    # 查看 quantile 的梯度与当前值
    print("当前 quantile (sigmoid(logit_q)) =", float(ups.quantile().detach().cpu().numpy()))
    if ups.logit_q.grad is not None:
        print("logit_q.grad =", float(ups.logit_q.grad.detach().cpu().numpy()))
    else:
        print("logit_q.grad is None")

    # 执行一步优化
    opt.step()

    print("更新后 quantile =", float(ups.quantile().detach().cpu().numpy()))

    # 打印 info 示例
    print("info sample:", {k: v[:5] for k, v in info.items()})

    # 说明: 在真实训练中，你应该将 DopplerUpsampler 嵌入你的模型（比如作为 model 的一部分），
    # 这样 optimizer = torch.optim.SGD(model.parameters(), ...) 会自动把 quantile 参数包含在内。
