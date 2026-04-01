from torch import nn
import torch
import torch.nn.functional as F

# modified from https://github.com/ToTheBeginning/PuLID
# https://github.com/tencent-ailab/IP-Adapter

def hack_unet_attn_layers(unet, CustomAttnProcessor):
    au_adapter_attn_procs = {}
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is not None:
            au_adapter_attn_procs[name] = CustomAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(unet.device)
        else:
            au_adapter_attn_procs[name] = AttnProcessor()
    unet.set_attn_processor(au_adapter_attn_procs)


class AttnProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        au_embedding=None,
        au_scale=1.0,
        image_embedding=None,
        ip_scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class AUAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None):
        super().__init__()
        self.au_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.au_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        # 🔥 注册为可学习参数
        # 初始温度设为 1.0
        self.temperature = nn.Parameter(torch.ones(1))
        # 🔥 修改：将 zeros(1) 换成 1e-4，打破梯度死锁！
        self.au_gate = nn.Parameter(torch.tensor([1e-4]))

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        au_embedding=None,
        au_scale=1.0,
    ):
        # residual = hidden_states

        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        # input_ndim = hidden_states.ndim

        # if input_ndim == 4:
        #     batch_size, channel, height, width = hidden_states.shape
        #     hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # batch_size, sequence_length, _ = (
        #     hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # )
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # if attn.group_norm is not None:
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query = attn.to_q(hidden_states)

        # if encoder_hidden_states is None:
        #     encoder_hidden_states = hidden_states
        # elif attn.norm_cross:
        #     encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # key = attn.to_k(encoder_hidden_states)
        # value = attn.to_v(encoder_hidden_states)

        # query = attn.head_to_batch_dim(query)
        # key = attn.head_to_batch_dim(key)
        # value = attn.head_to_batch_dim(value)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = attn.batch_to_head_dim(hidden_states)

        # for au-adapter
        # au_key = self.au_to_k(au_embedding)
        # au_value = self.au_to_v(au_embedding)

        # au_key = attn.head_to_batch_dim(au_key).to(query.dtype)
        # au_value = attn.head_to_batch_dim(au_value).to(query.dtype)

        # au_attention_probs = attn.get_attention_scores(query, au_key, None)
        # au_hidden_states = torch.bmm(au_attention_probs, au_value)
        # au_hidden_states = attn.batch_to_head_dim(au_hidden_states)

        # hidden_states = hidden_states + au_scale * au_hidden_states

        # # linear proj
        # hidden_states = attn.to_out[0](hidden_states)
        # # dropout
        # hidden_states = attn.to_out[1](hidden_states)

        # if input_ndim == 4:
        #     hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     hidden_states = hidden_states + residual

        # hidden_states = hidden_states / attn.rescale_output_factor

        # return hidden_states

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # =========================================================
        # 创新点 1：基于 Sigmoid 的空间精准注意力 (Spatial Mask)
        # =========================================================
        au_key = self.au_to_k(au_embedding)
        au_value = self.au_to_v(au_embedding)

        au_key = attn.head_to_batch_dim(au_key).to(query.dtype)
        au_value = attn.head_to_batch_dim(au_value).to(query.dtype)

        # 1. 计算 attention score
        au_attention_scores = torch.bmm(query, au_key.transpose(-1, -2))
        
        # 2. 归一化
        au_attention_scores = au_attention_scores * attn.scale
        
        # 3. 加温度 (加 abs() 防止学出负数导致逻辑反转)
        au_attention_scores = au_attention_scores / (self.temperature.abs() + 1e-6)
        
        # 4. Sigmoid gating
        spatial_mask = torch.sigmoid(au_attention_scores)
        
        # =========================================================
        # 创新点：Gaussian Face Prior (面部高斯先验引导)
        # =========================================================
        seq_len = spatial_mask.shape[1] 
        side_len = int(seq_len ** 0.5)
        
        # 确保当前层是一个正方形的特征图（扩散模型里通常是 64x64, 32x32, 16x16）
        if side_len * side_len == seq_len:
            # 动态生成二维高斯分布 (为了不拖慢训练，只有第一步时计算，之后缓存复用)
            if not hasattr(self, 'face_prior') or getattr(self, 'face_prior').shape[0] != seq_len:
                device = query.device
                
                # 生成坐标网格 (范围从 -1 到 1)
                y, x = torch.meshgrid(
                    torch.linspace(-1, 1, side_len, device=device), 
                    torch.linspace(-1, 1, side_len, device=device), 
                    indexing='ij'
                )
                
                # 计算二维高斯分布。
                # sigma=0.55 是一个经验值：刚好覆盖人脸的五官，并让边角的权重迅速衰减为 0
                gaussian = torch.exp(-(x**2 + y**2) / (2 * 0.55**2))
                
                # 展平为 (seq_len, 1)，比如 (4096, 1)，并对齐数据类型
                self.face_prior = gaussian.view(-1, 1).to(query.dtype)
            
            # 【绝杀操作】：将网络算出来的 mask 直接乘上高斯先验！
            # spatial_mask 形状为 (Batch*Heads, 4096, 12)
            # self.face_prior 形状为 (4096, 1)
            # PyTorch 会自动在最后一个维度广播，让 12 个 AU 同时被屏蔽掉背景
            spatial_mask = spatial_mask * self.face_prior
        # =========================================================
        
        # 5. Residual gating (依然保留一点点底噪，防止死锁)
        spatial_mask = spatial_mask * 0.9 + 0.1
        
       # ====== 内存挂载：提取 12通道 Mask 供 train.py 可视化 ======
        if side_len * side_len == seq_len and side_len == 64:
            # 此时 spatial_mask 形状为 (Batch*Heads, 4096, 12)
            # 使用 .detach().clone() 安全剥离计算图
            self.current_spatial_mask = spatial_mask.detach().clone()
        # =========================================================
        
        # 6. 提取特征
        au_hidden_states = torch.bmm(spatial_mask, au_value)
        
        # 6. 提取特征
        au_hidden_states = torch.bmm(spatial_mask, au_value)
        au_hidden_states = attn.batch_to_head_dim(au_hidden_states)

        # 7. 注入 (配合 Zero-Gating，平滑起步)
        hidden_states = hidden_states + au_scale * self.au_gate * au_hidden_states
        # =========================================================

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AUIPAttnProcessor2_0(torch.nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.au_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.au_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        au_embedding=None,
        au_scale=1.0,
        image_embedding=None,
        ip_scale=1.0,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(image_embedding)
        ip_value = self.to_v_ip(image_embedding)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        # for au adapter
        au_key = self.au_to_k(au_embedding)
        au_value = self.au_to_v(au_embedding)

        au_key = au_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        au_value = au_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        au_hidden_states = F.scaled_dot_product_attention(
            query, au_key, au_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        au_hidden_states = au_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        au_hidden_states = au_hidden_states.to(query.dtype)

        hidden_states = hidden_states + ip_scale * ip_hidden_states + au_scale * au_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
