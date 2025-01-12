# model/transformer_model.py
import torch
import torch.nn as nn
import math
from config import DIM, ENC_LAYER, DEC_LAYER, HEAD, NUM_CLASSES, HEAD, FF_MULTIPLIER, DROPOUT, FEATURE, D_MODEL, DIM, DROPOUT, PHAZE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :].to(x.device)

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=HEAD, dim_feedforward=DIM, num_splits=4, dropout=DROPOUT):
        super(CustomTransformerDecoderLayer, self).__init__()

        # 第一层自注意力
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # Add & Norm 层
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 第二层交叉注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # Add & Norm 层
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 前馈层
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

        # Add & Norm 层
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.num_splits = num_splits
        self.split_layer = nn.Linear(d_model, d_model * num_splits)
        # 可训练的权重，用于对 cross_attn 结果加权
        self.attention_weights = nn.Parameter(torch.ones(num_splits))

    def forward(self, short_history, encoded, key_padding_mask):
        # 第一层自注意力层（短历史的自注意力）
        short_history = short_history.permute(1, 0, 2)
        self_attn_output, _ = self.self_attn(short_history, short_history, short_history)

        short_history = short_history + self.dropout1(self_attn_output)
        short_history = self.norm1(short_history)
        # 第二层交叉注意力（编码器与短历史的交互）
        #key_padding_mask = key_padding_mask.permute(1, 0, 2)
        encoded = encoded.permute(1, 0, 2)
        cross_attn_output, _ = self.cross_attn(short_history, encoded, encoded)

        short_history = short_history + self.dropout2(cross_attn_output)
        short_history = self.norm2(short_history)

        ff_output = self.feedforward(short_history)
        short_history = short_history + self.dropout3(ff_output)
        short_history = self.norm3(short_history)

        output = short_history.permute(1, 0, 2)

        return output

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=FEATURE, d_model=D_MODEL, enc_layers=ENC_LAYER, dec_layers=DEC_LAYER, num_heads=HEAD,
                 num_classes=NUM_CLASSES, dropout=DROPOUT,dim_feedforward=DIM,  ff_multiplier=FF_MULTIPLIER):
        super(TransformerClassifier, self).__init__()

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=enc_layers)

        self.decoder = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(dec_layers)
        ])

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.saved_long_history = None
        self.saved_encoded = None
        self.phaze = PHAZE
        self.i = 0

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier (Glorot) initialization for Linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                # Kaiming (He) initialization for Convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,
                                                                                                      nn.ConvTranspose3d):
                # Xavier initialization for Transposed Convolutional layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                # Initialize BatchNorm with weights close to 1, biases close to 0
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Embedding):
                # Normal initialization for Embedding layers
                nn.init.normal_(m.weight, mean=0, std=0.01)

            elif isinstance(m, nn.LayerNorm):
                # Constant initialization for LayerNorm layers
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

            elif isinstance(m, nn.RNNBase) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                # Orthogonal initialization for recurrent weights
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

            elif hasattr(m, 'reset_parameters'):
                # If the layer has a reset_parameters function, use it as default
                m.reset_parameters()

    def forward(self, long_history, short_history, long_history_mask=None, tgt_mask=None):

        if self.phaze is 'train':
        #print(long_history.shape, short_history.shape)
            self.saved_long_history = long_history
            long_history = long_history.view(-1, long_history.size(2), long_history.size(3))
            long_history = long_history.permute(1, 0, 2)

            long_history_mask = long_history_mask.view(-1, long_history_mask.size(2))
            long_history_mask = long_history_mask.T

            long_history = self.input_embedding(long_history)

            long_history = self.positional_encoding(long_history.permute(1, 0, 2))

            encoded = self.encoder(long_history, src_key_padding_mask=long_history_mask)
            self.saved_encoded = encoded
        else:
            if self.saved_encoded is None or not torch.equal(self.saved_long_history, long_history):
                print("saved" , self.i)
                self.saved_long_history = long_history
                long_history = long_history.view(-1, long_history.size(2), long_history.size(3))
                long_history = long_history.permute(1, 0, 2)

                long_history_mask = long_history_mask.view(-1, long_history_mask.size(2))
                long_history_mask = long_history_mask.T

                long_history = self.input_embedding(long_history)

                long_history = self.positional_encoding(long_history.permute(1, 0, 2))

                encoded = self.encoder(long_history, src_key_padding_mask=long_history_mask)
                self.saved_encoded = encoded


        encoded = self.saved_encoded
        self.i = self.i + 1
        #print("encoded shape", encoded.shape)
       #decoded = self.decoder(short_history, encoded, tgt_key_padding_mask=tgt_mask)
        short_history = short_history.view(-1, short_history.size(2), short_history.size(3))
        short_history = short_history.permute(1, 0, 2)
        short_history = self.input_embedding(short_history)
        short_history = self.positional_encoding(short_history.permute(1, 0, 2))

        decoded = short_history

        for layer in self.decoder:
            decoded = layer(decoded, encoded, long_history_mask)

        #print("decoded.shape" ,decoded.shape)
        decoded = decoded.reshape(-1, decoded.size(-1))  # 调整为 (30*10, 128)
        output = self.fc(decoded)  # 输出形状为 (300, 2)
        #print("output shape", output.shape)
        #output = output.reshape(2, 30, -1)
        return output



        '''
         if torch.isnan(short_history).any():
            print("NaN detected in batch_short_histories")
        else:
            print("No NaN in batch_short_histories")

        if torch.isnan(encoded).any():
            print("NaN detected in batch_long_histories")
        else:
            print("No NaN in batch_long_histories")

        if key_padding_mask.dtype != torch.bool:
            print("Non-bool values detected in batch_long_masks")
        else:
            print("All values in batch_long_masks are boolean")
        '''
        '''
                self.decoder_layer = nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation='gelu'
                )
                self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=dec_layers)
                '''