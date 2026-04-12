from torch import nn
import torch
import torch.nn.functional as F

# modified from https://github.com/ToTheBeginning/PuLID
# https://github.com/tencent-ailab/IP-Adapter

def hack_unet_attn_layers(unet, CustomAttnProcessor=None):
    # 这里做个兼容，如果没有传，默认使用我们自己强大的 AUAttnProcessor
    if CustomAttnProcessor is None:
        CustomAttnProcessor = AUAttnProcessor
        
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
    # (保持原有不变，处理自注意力)
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

        # =========================================================================
        # [创新点 1]: 空锚点吸收器 (The Null-Token Absorber)
        # 初始化一个长度相同的随机 Token，供背景像素“倾泻”它们被迫分配的注意力
        # =========================================================================
        self.null_token = nn.Parameter(torch.randn(1, 1, cross_attention_dim or hidden_size) * 0.02)

        # =========================================================================
        # [创新点 2]: 零初始化的通道级动态门控 (Zero-Init Channel Gating)
        # 保护底层先验，实现精细的通道筛选
        # =========================================================================
        self.gamma = nn.Parameter(torch.zeros(hidden_size))

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

        # ------------------- 1. 原生文本交叉注意力 -------------------
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

        # ------------------- 2. AU 融合：空锚点吸收与通道门控 -------------------
        target_dtype = query.dtype
        
        # a. 拼接第 13 个 Null-Token
        expanded_null = self.null_token.expand(batch_size, -1, -1).to(target_dtype)
        # extended_au_embedding shape: (Batch, 13, 1024)
        extended_au_embedding = torch.cat([au_embedding, expanded_null], dim=1)

        # b. 映射到 Key 和 Value
        au_key = self.au_to_k(extended_au_embedding)
        au_value = self.au_to_v(extended_au_embedding)
        
        # c. 【极其关键的物理切断】：强行让第 13 个 Token 的 Value 变成全 0 向量！
        # 这样背景区域即便把 99% 的注意力分给了它，最后乘出来提取到的特征也只是纯净的 0。
        au_value[:, -1, :] = 0.0

        au_key = attn.head_to_batch_dim(au_key).to(target_dtype)
        au_value = attn.head_to_batch_dim(au_value).to(target_dtype)

        # d. 计算注意力分数
        attention_scores = torch.bmm(query, au_key.transpose(-1, -2)) * attn.scale
        au_attention_probs = attention_scores.softmax(dim=-1).to(target_dtype)

        # e. 【新增】：剥离提取用于可视化的两种 Mask
        if not self.training or torch.rand(1).item() < 0.05:
            head_dim_count = query.shape[0] // batch_size
            avg_probs = au_attention_probs.view(batch_size, head_dim_count, -1, 13).mean(dim=1) # (B, seq_len, 13)
            
            # 真实 AU 面部形变区 (前 12 个 Token 概率之和)
            attn.current_spatial_mask = avg_probs[..., :12].sum(dim=-1).detach().cpu()
            # 隔离吸收区 (第 13 个 Null-Token 的概率)
            attn.current_null_mask = avg_probs[..., 12].detach().cpu()

        # f. 提取 AU 特征
        au_hidden_states = torch.bmm(au_attention_probs, au_value)
        au_hidden_states = attn.batch_to_head_dim(au_hidden_states).to(target_dtype)

        # g. 【新增】：Zero-Init 通道级动态门控融合
        modulated_au_states = self.gamma.to(target_dtype) * au_hidden_states
        hidden_states = hidden_states + (au_scale * modulated_au_states)

        # ------------------- 3. 扫尾工作 -------------------
        hidden_states = attn.to_out[0](hidden_states)
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
