from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        # print(v)    # torch.Size([64, 14, 512])
        # print(k)    # torch.Size([64, 14, 512])
        # print(q)    # torch.Size([64, 14, 512])
        # print(mask) # torch.Size([64, 1, 1, 14])

        # v = self.linear_v(v)
        # v = v.view(n_batches, -1, self.__C.MULTI_HEAD, self.__C.HIDDEN_SIZE_HEAD)
        # v = v.transpose(1, 2)

        v = self.linear_v(v).view(  # reshape tensor to 64 x 14 x 8 x 64
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(    # view() requires contiguous Tensor
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        # print(value.shape)  # torch.Size([64, 8, 14, 64])
        # print(key.shape)    # torch.Size([64, 8, 14, 64])
        # print(query.shape)  # torch.Size([64, 8, 14, 64])
        # print(mask.shape)   # torch.Size([64, 1, 1, 14])

        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(
            x + self.dropout1(self.mhatt(x, x, x, x_mask))
        )
        x = self.norm2(
            x + self.dropout2(self.ffn(x))
        )
        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    # def forward(self, x, y, x_mask, y_mask):
    #     x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))
    #     x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
    #     x = self.norm3(x + self.dropout3(self.ffn(x)))
    #     return x

    def forward(self, y, x, y_mask, x_mask):
        y = self.norm1(y + self.dropout1(self.mhatt1(y, y, y, y_mask)))
        y = self.norm2(y + self.dropout2(self.mhatt2(x, x, y, x_mask)))
        y = self.norm3(y + self.dropout3(self.ffn(y)))
        return y

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()
        # __C.LAYER = 6
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])  # SA x 6
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)]) # SGA x 6
        # print(self.enc_list)
        # print(self.dec_list)

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)
        return x, y

class MCA_STACK(nn.Module):
    def __init__(self, __C):
        super(MCA_STACK, self).__init__()
        self.LAYER = __C.LAYER
        self.SA_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.SGA_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        for i in range(self.LAYER):
            x = self.SA_list[i](x, x_mask)
            y = self.SGA_list[i](y, x, y_mask, x_mask)
        return x, y


