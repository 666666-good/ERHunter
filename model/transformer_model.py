# model/transformer_model.py
import torch
import torch.nn as nn
import math
import torch.fft as fft
import numpy as np
from config import ENC_LAYER, HEAD, NUM_CLASSES, HEAD, FEATURE, D_MODEL, FF_DIM, DROPOUT, PHAZE, SEQ_LENGTH_TRANS,\
    IN_CHANNEL, OUT_CHANNEL
import matplotlib.pyplot as plt
from PIL import Image
import os
from matplotlib import rcParams

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

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
        #print(index)
    else:
        index = list(range(0, modes))
    index.sort()
    return index

class FourierCrossAttention(nn.Module):
    def __init__(self, modes=64, mode_select_method='random', activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = IN_CHANNEL
        self.out_channels = OUT_CHANNEL
        self.seq_len_q = SEQ_LENGTH_TRANS
        self.seq_len_kv = SEQ_LENGTH_TRANS
        self.d_model = D_MODEL
        self.num_heads = HEAD
        self.index_q = get_frequency_modes(self.seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(self.seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, self.in_channels // 8, self.out_channels // 8, len(self.index_q), dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(8, self.in_channels // 8, self.out_channels // 8, len(self.index_q), dtype=torch.float))

        self.wq = nn.Linear(self.d_model // self.num_heads, self.d_model // self.num_heads)
        self.wk = nn.Linear(self.d_model // self.num_heads, self.d_model // self.num_heads)
        self.wv = nn.Linear(self.d_model // self.num_heads, self.d_model // self.num_heads)

        '''
        self.in_proj_weight = nn.Parameter(torch.empty(3 * self.d_model, self.d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * self.d_model))
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        '''
    # Complex multiplication
    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v):
        '''
        # size = [B, L, H, E]
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        _b = self.in_proj_bias
        _start = None
        _end = embed_dim
        _w = self.in_proj_weight[:_end, :]
        if _b is not None:
            _b = _b[:_end]
        q = nn.linear(q, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = nn.linear(k, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = nn.linear(v, _w, _b)
        '''
        #x = x.view(batch_size, sequence_length, d_model)

        '''
        x_ft = torch.fft.rfft(q, dim=1)  # 在 sequence_length 维度进行傅里叶变换
        x_magnitude = torch.abs(x_ft).mean(dim=(0, 2, 3))  # 对 batch_size, num_heads, embedding_dim 维度求平均
        config = {
            "font.family": 'Arial',
            "font.size": 7
        }
        rcParams.update(config)

        plt.figure(figsize=(7, 5))
        plt.subplots_adjust(left=0.4, right=0.55, top=0.6, bottom=0.4)

        plt.plot(x_magnitude.cpu().numpy())
        #plt.title("Frequency Spectrum of 'x'")
        plt.xlabel("Frequency component")
        plt.ylabel("Energy")
        plt.grid()
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        image_path = os.path.join(desktop_path, "x_frequency_spectrum.eps")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        print(f"频谱图已保存到: {image_path}")
        '''

        q = self.wq(q)  # 形状仍然是 (batch_size, sequence_length, num_heads, embedding_dim)
        k = self.wk(k)  # 形状仍然是 (batch_size, sequence_length, num_heads, embedding_dim)
        v = self.wv(v)
        #print(q.shape, k.shape, v.shape)

        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)

        '''
        xq_magnitude = torch.abs(xq_ft).mean(dim=(0, 1, 2))  # 对 batch_size、num_heads 和 embedding_dim 平均
        plt.figure(figsize=(10, 6))
        plt.plot(xq_magnitude.cpu().numpy())
        plt.title("Frequency Spectrum of 'xq_ft'")
        plt.xlabel("Frequency Index")
        plt.ylabel("Magnitude")
        plt.grid()
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        image_path = os.path.join(desktop_path, "xq_frequency_spectrum.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        '''

        for i, j in enumerate(self.index_q):
            if j >= xq_ft.shape[3]:
                continue
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        '''
        out_magnitude = torch.abs(xq_ft_).mean(dim=(0, 1, 2))  # 对 batch_size、num_heads 和 embedding_dim 平均
        plt.figure(figsize=(10, 6))
        plt.plot(out_magnitude.cpu().numpy())
        plt.title("Frequency Spectrum of 'out'")
        plt.xlabel("Frequency Index")
        plt.ylabel("Magnitude")
        plt.grid()
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        image_path = os.path.join(desktop_path, "frequency.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        '''
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            if j >= xk_ft.shape[3]:
                continue
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
            #print(xk_ft_[:, :, :, i])

        # perform attention mechanism on frequency domain
        xqk_ft = (self.compl_mul1d("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = self.compl_mul1d("bhxy,bhey->bhex", xqk_ft, xk_ft_)
       # print("xqkv_ft shape", xqkv_ft.shape[0], xqkv_ft.shape[1], xqkv_ft.shape[2], xqkv_ft.shape[3])
        xqkvw = self.compl_mul1d("bhex,heox->bhox", xqkv_ft, torch.complex(self.weights1, self.weights2))
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        #print("xqkw shape", xqkvw.shape[0], xqkvw.shape[1], xqkvw.shape[2], xqkvw.shape[3])
        #print("out_ft shape", out_ft.shape[0], out_ft.shape[1], out_ft.shape[2], out_ft.shape[3])
        for i, j in enumerate(self.index_q):
            if i >= xqkvw.shape[3] or j >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        #print(out_ft.shape)

        '''
        out_magnitude = torch.abs(out_ft).mean(dim=(0, 1, 2))  # 对 batch_size、num_heads 和 embedding_dim 平均
        plt.figure(figsize=(10, 6))
        plt.plot(out_magnitude.cpu().numpy())
        plt.title("Frequency Spectrum of 'out'")
        plt.xlabel("Frequency Index")
        plt.ylabel("Magnitude")
        plt.grid()
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        image_path = os.path.join(desktop_path, "frequency.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        '''

        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)

class UNet1D(nn.Module):
    def __init__(self, in_channels = 16, out_channels = 128):
        super(UNet1D, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        x = x.permute(0, 2, 1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        # 瓶颈部分
        bottleneck = self.bottleneck(self.maxpool(enc4))

        # 解码器部分
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # 跳跃连接
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)  # 跳跃连接
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)  # 跳跃连接
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # 跳跃连接
        dec1 = self.decoder1(dec1)

        # 最终输出
        out = self.final_conv(dec1)
        return out

def conv_block(in_channels, out_channels, kernel_size, padding, stride, groups):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups),
        nn.ReLU()
    )

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        #self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups = self.groups)  # 分组卷积
        #self.conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, groups=2)
        #self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, groups=2)
        self.conv1 = conv_block(16, 32, 3,1, 1, 1)  # 输入 16 通道，输出 32 通道
        self.conv2 = conv_block(32, 64, 3,1, 1, 1)  # 输入 32 通道，输出 64 通道
        #self.conv21 = conv_block(64, 64, 3, 1, 1, 1)  # 输入 32 通道，输出 64 通道
        self.conv3 = conv_block(64, 128,5,2, 1, 2)  # 输入 64 通道，输出 128 通道

        self.feature = FEATURE
        #self.fc = nn.Linear(12800, 2)  # 假设是二分类任务
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        #print("x.shape",x.size())
        x = self.conv1(x)  # 第一层卷积
        x = self.pool(x)

        x = self.conv2(x)  # 第一层卷积
        x = self.pool(x)

        #x = self.conv21(x)  # 第一层卷积

        x = self.conv3(x)  # 第一层卷积
        x = self.pool(x)

        # print(x.shape)
        #x = x.view(x.size(0), -1)  # 展平为一维张量
        #print("x.shape", x.shape)
        #x = self.fc(x)  # 全连接层输出
        #print("x.shape", x.shape)

        '''
        x = x.reshape(100, 128)
        x = x.to(torch.float32)
        grayscale_values = x.mean(axis=1)
        grayscale_values_np = grayscale_values.numpy()
        grayscale_normalized = (grayscale_values_np - grayscale_values_np.min()) / (
                grayscale_values_np.max() - grayscale_values_np.min()) * 255
        grayscale_normalized = grayscale_normalized.astype(np.uint8)
        inverted_grayscale = 255 - grayscale_normalized
        grayscale_matrix = inverted_grayscale.reshape(10, 10)
        plt.imshow(grayscale_matrix, cmap='gray')
        plt.axis('off')
        #plt.title("10x10 Grayscale Image from 16D Data")
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        image_path = os.path.join(desktop_path, "grayscale_10x10_image.eps")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        exit()
        '''

        #print(x.shape)
        #print("x.shape",x.shape)
        #x = x.view(x.size(0), -1)  # 展平为一维张量
        #x = self.fc(x)  # 全连接层输出

        return x

# Transformer Model with multi-resolution adjustments
class FourierTransformer(nn.Module):
    def __init__(self):
        super(FourierTransformer, self).__init__()

        self.in_channels = IN_CHANNEL
        self.out_channels = OUT_CHANNEL
        self.num_heads = HEAD
        self.num_layers = ENC_LAYER
        self.num_classes = NUM_CLASSES
        self.d_model = D_MODEL
        self.feature = FEATURE
        self.ff_hid_dim = FF_DIM
        self.dropout = nn.Dropout(DROPOUT)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.embedding = nn.Linear(self.feature, self.d_model)
        # Position encoding
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.attention_layers = nn.ModuleList([
            FourierCrossAttention()
            for _ in range(self.num_layers)
        ])

        self.norm_layers1 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.d_model),
            )
            for _ in range(self.num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.ff_hid_dim),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(self.ff_hid_dim, self.d_model),
            )
            for _ in range(self.num_layers)
        ])

        self.norm_layers2 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.d_model),
            )
            for _ in range(self.num_layers)
        ])
        #self.embedding = nn.Linear(16, 128)
        # Output layer for classification (or other tasks)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x):
        #print(x.shape)
        x = x.permute(0, 2, 1)

        '''
        x = x.reshape(-1, 16)
        x = self.embedding(x)
        #print(x.shape)
        x =x.reshape(64, 100, 128)
        '''
        #print(x.shape)
        x = self.positional_encoding(x) #batch_size, sequence_length, d_model
        batch_size, sequence_length, d_model = x.shape
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        embedding_dim = d_model // self.num_heads

        #q = q * scaling
        #print(x.size())
        for i in range(self.num_layers):
            # Fourier Cross Attention
            x = x.view(batch_size, sequence_length, self.num_heads, embedding_dim)
            #x = x.view(batch_size, sequence_length, d_model)
            x2, _ = self.attention_layers[i](x, x, x)
            x2 = x2.reshape(x.shape)
            x = x + self.dropout1(x2)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.norm_layers1[i](x)
            x2 = self.ffn_layers[i](x)
            x = x + self.dropout2(x2)
            x = self.norm_layers2[i](x)

        # Step 3: Output classification (for example, for a classification task)
        x = x.mean(dim=1)  # Global average pooling over sequence length
        x = self.classifier(x)
        #print(x.size())
        #exit()
        return x

class ERHunter(nn.Module):
    def __init__(self):
        super(ERHunter, self).__init__()
        self.cnn = CustomCNN()
        self.fourier_transformer = FourierTransformer()

    def forward(self, x):
        out = self.cnn(x)
        #print(out.shape)
        out = self.fourier_transformer(out)
        return out

'''
        x = self.conv1(x)  # 第一层卷积
        x = self.relu(x)
        x = self.pool(x)  # 池化层

        x = self.conv2(x)  # 第二层卷积
        x = self.relu(x)
        x = self.pool(x)  # 池化层

        x = self.conv3(x)  # 第三层卷积
        x = self.relu(x)
        #print(x.shape)
        x = self.pool(x)  # 池化层
        '''