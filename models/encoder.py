
# import torch
# import torch.nn as nn

# class BLSTMEncoder(nn.Module):
#     def __init__(self, input_size=40, hidden_size=256, num_layers=4, dropout=0.3):
#         """
#         input_size  : number of mel bins (40)
#         hidden_size : LSTM hidden units per direction
#         num_layers  : number of stacked BLSTM layers
#         dropout     : dropout between layers
#         """
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,       # input shape: (batch, frames, features)
#             bidirectional=True,
#             dropout=dropout if num_layers > 1 else 0.0
#         )
#         self.output_size = hidden_size * 2  # ×2 because bidirectional

#     def forward(self, x):
#         """
#         x   : (batch, frames, input_size)
#         out : (batch, frames, hidden_size * 2)
#         """
#         out, _ = self.lstm(x)
#         return out


# %%writefile /home/ju1ian/Documents/eend/models/encoder.py

import torch
import torch.nn as nn

class BLSTMEncoder(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Thêm LayerNorm để ổn định huấn luyện
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.output_size = hidden_size * 2

    def forward(self, x):
        # x : (batch, frames, input_size)
        out, _ = self.lstm(x)
        out = self.ln(out) 
        return out
