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

        self.hidden_size = hidden_size

        # === AU → K,V 投影 ===
        self.au_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.au_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        # === Spatial Modulation Gate（FiLM-like） ===
        self.au_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # 🔥 更合理初始化（不会一开始全关）
        nn.init.zeros_(self.au_gate[2].weight)
        nn.init.constant_(self.au_gate[2].bias, -0.5)

        # === Temperature（控制mask sharpness）===
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

        # === 可选：mask 正则缓存 ===
        self.current_spatial_mask = None

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

        # ------------------- 0. 预处理 -------------------
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        B, N, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # ------------------- 1. 原始 Cross-Attention（text/image） -------------------
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

        attn_probs = attn.get_attention_scores(query, key, attention_mask)
        base_hidden_states = torch.bmm(attn_probs, value)
        base_hidden_states = attn.batch_to_head_dim(base_hidden_states)

        # ------------------- 2. ⭐ Spatial Routing（核心） -------------------

        # === AU → K,V ===
        target_dtype = query.dtype
        au_key = self.au_to_k(au_embedding)
        au_value = self.au_to_v(au_embedding)

        au_key = attn.head_to_batch_dim(au_key).to(target_dtype)
        au_value = attn.head_to_batch_dim(au_value).to(target_dtype)

        # === Attention Score（Q × AU）===
        # (B*H, HW, 12)
        attention_scores = torch.bmm(query, au_key.transpose(-1, -2))

        # 🔥 关键1：转成 AU → spatial
        # (B*H, 12, HW)
        attention_scores = attention_scores.transpose(1, 2)

        # 🔥 关键2：temperature（正确用法）
        attention_scores = attention_scores / self.temperature.clamp(min=1e-4)

        # === Spatial Mask ===
        au_attention_probs = attention_scores.softmax(dim=-1)  # over spatial

        # ------------------- 3. 保存 mask（论文可视化） -------------------
        if not self.training or torch.rand(1).item() < 0.05:
            num_heads = query.shape[0] // B
            mask = au_attention_probs.view(B, num_heads, -1, au_attention_probs.shape[-1])
            mask = mask.mean(dim=1)  # (B, 12, HW)
            self.current_spatial_mask = mask.detach().cpu()

        # ------------------- 4. AU Feature 提取 -------------------

        # (B*H, 12, HW) × (B*H, 12, dim)
        au_hidden_states = torch.bmm(
            au_attention_probs.transpose(1, 2),  # (B*H, HW, 12)
            au_value
        )

        au_hidden_states = attn.batch_to_head_dim(au_hidden_states)

        # 🔥 防止 AU 太弱
        au_hidden_states = au_hidden_states * 2.0

        # ------------------- 5. Spatial Modulation（FiLM-like） -------------------

        gate_dtype = next(self.au_gate.parameters()).dtype

        gate = torch.sigmoid(
            self.au_gate(au_hidden_states.to(gate_dtype))
        ).to(target_dtype)

        # 🔥 residual-style modulation（更稳定）
        modulated_au = gate * au_hidden_states

        # ------------------- 6. 融合 -------------------

        hidden_states = base_hidden_states + au_scale * modulated_au

        # ------------------- 7. 输出 -------------------
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

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
