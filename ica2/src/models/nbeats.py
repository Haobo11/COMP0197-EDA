"""N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting.

参考: Oreshkin et al., "N-BEATS: Neural basis expansion analysis
       for interpretable time series forecasting", ICLR 2020

本实现为 **Generic** 架构（可学习基函数），并扩展为概率输出 (mu, var)
以兼容项目的 Gaussian NLL 训练流程。
"""

import torch
import torch.nn as nn

from .base import BaseModel
from . import register_model


class NBeatsBlock(nn.Module):
    """Generic N-BEATS 块。

    数据流: x (flat) -> FC stack -> theta
        -> backcast_fc(theta)  →  重构输入（用于残差）
        -> forecast_fc(theta)  →  预测贡献（用于累加）
    """

    def __init__(self, input_dim, fc_dim, forecast_dim,
                 num_fc_layers=4, dropout=0.0):
        super().__init__()

        # 全连接栈
        layers = []
        for i in range(num_fc_layers):
            in_d = input_dim if i == 0 else fc_dim
            layers.append(nn.Linear(in_d, fc_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*layers)

        # Generic 基函数：可学习线性映射（无 bias，与原文一致）
        self.backcast_fc = nn.Linear(fc_dim, input_dim, bias=False)
        self.forecast_fc = nn.Linear(fc_dim, forecast_dim, bias=False)

    def forward(self, x):
        """
        x: (batch, input_dim)
        返回: backcast (batch, input_dim), forecast (batch, forecast_dim)
        """
        theta = self.fc(x)
        return self.backcast_fc(theta), self.forecast_fc(theta)


@register_model("nbeats")
class TimeSeriesNBeats(BaseModel):
    """基于 N-BEATS (Generic) 的时序预测模型。

    数据流: 输入(batch, seq_len, n_features)
        -> 展平为 (batch, seq_len * n_features)
        -> [NBeatsBlock × (num_stacks × num_blocks_per_stack)]
           每块: residual -= backcast; forecast += block_forecast
        -> mu_head / logvar_head -> (mu, var)

    与 Transformer / Mamba / LSTM 保持相同的输入输出接口。
    """

    def __init__(self, n_features, seq_len,
                 num_stacks=2, num_blocks_per_stack=3,
                 num_fc_layers=4, fc_dim=256,
                 forecast_dim=64, dropout=0.5):
        super().__init__()
        self.forecast_dim = forecast_dim
        input_dim = seq_len * n_features

        # 所有块按顺序排列，块间通过 backcast 残差连接
        total_blocks = num_stacks * num_blocks_per_stack
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_dim, fc_dim, forecast_dim,
                        num_fc_layers, dropout)
            for _ in range(total_blocks)
        ])

        # 概率输出头
        self.mu_head = nn.Sequential(
            nn.Linear(forecast_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(forecast_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        返回: (mu, var)，各 shape 为 (batch,)
        """
        batch = x.size(0)

        # 展平：(batch, seq_len * n_features)
        residual = x.reshape(batch, -1)

        # 双残差机制：backcast 逐块减去，forecast 逐块累加
        forecast = torch.zeros(batch, self.forecast_dim, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        # 概率输出
        mu = self.mu_head(forecast).squeeze(-1)
        log_var = self.logvar_head(forecast).squeeze(-1)
        var = torch.exp(log_var)
        return mu, var

    @classmethod
    def from_config(cls, model_cfg, n_features):
        return cls(
            n_features=n_features,
            seq_len=model_cfg["seq_len"],
            num_stacks=model_cfg.get("num_stacks", 2),
            num_blocks_per_stack=model_cfg.get("num_blocks_per_stack", 3),
            num_fc_layers=model_cfg.get("num_fc_layers", 4),
            fc_dim=model_cfg.get("fc_dim", 256),
            forecast_dim=model_cfg.get("forecast_dim", 64),
            dropout=model_cfg.get("dropout", 0.5),
        )
