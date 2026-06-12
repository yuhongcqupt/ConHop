import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv, SAGEConv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout_ratio):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
            h_list.append(h)

        return h, h_list


class GCN(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout_ratio, activation):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(in_dim, out_dim, activation=activation))
        else:
            self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(GraphConv(hidden_dim, out_dim, activation=activation))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.norms[l](h)
                h = self.dropout(h)
            h_list.append(h)

        return h, h_list


class GAT(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        hidden_dim,
        out_dim,
        dropout_ratio,
        activation,
        num_heads=8,
        attn_drop=0.3,
        negative_slope=0.2,
        residual=False,
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        hidden_per_head = hidden_dim // num_heads
        heads = ([num_heads] * (num_layers - 1)) + [1]

        if num_layers == 1:
            self.layers.append(
                GATConv(
                    in_dim,
                    out_dim,
                    1,
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    None,
                )
            )
        else:
            self.layers.append(
                GATConv(
                    in_dim,
                    hidden_per_head,
                    heads[0],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    False,
                    activation,
                )
            )
            for l in range(1, num_layers - 1):
                self.layers.append(
                    GATConv(
                        hidden_per_head * heads[l - 1],
                        hidden_per_head,
                        heads[l],
                        dropout_ratio,
                        attn_drop,
                        negative_slope,
                        residual,
                        activation,
                    )
                )
            self.layers.append(
                GATConv(
                    hidden_per_head * heads[-2],
                    out_dim,
                    heads[-1],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    None,
                )
            )

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
            h_list.append(h)

        return h, h_list


class GraphSAGE(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout_ratio, activation, aggregator_type='mean'):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.activation = activation

        if num_layers == 1:
            self.layers.append(SAGEConv(in_dim, out_dim, aggregator_type=aggregator_type))
        else:
            self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type=aggregator_type))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_type))
            self.layers.append(SAGEConv(hidden_dim, out_dim, aggregator_type=aggregator_type))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
            h_list.append(h)

        return h, h_list


def _sample_gumbel_like(logits, eps=1e-20):
    u = torch.rand_like(logits)
    return -torch.log(-torch.log(u + eps) + eps)


def build_sparse_gate(logits, topk=2, tau=1.0, mode='soft_topk_st', training=True):
    """
    logits: [N, H]
    返回:
        gate: [N, H]，行归一化后可直接做加权和

    支持三种模式:
        1) soft
           纯 softmax，不做 hard top-k。最稳，最容易复现。

        2) soft_topk_st
           直通估计的 hard top-k。工程默认，通常最稳。

        3) gumbel_topk_st
           训练时加入 Gumbel 噪声，增强离散选择探索；推理时退化为确定性 top-k。
    """
    if logits.dim() != 2:
        raise ValueError(f'Expected logits shape [N, H], got {tuple(logits.shape)}')

    num_hops = logits.size(1)
    topk = max(1, min(int(topk), num_hops))
    tau = max(float(tau), 1e-6)

    if mode == 'soft':
        gate = torch.softmax(logits / tau, dim=-1)
        return gate

    routed_logits = logits
    if mode == 'gumbel_topk_st' and training:
        routed_logits = routed_logits + _sample_gumbel_like(routed_logits)

    probs = torch.softmax(routed_logits / tau, dim=-1)

    top_idx = torch.topk(routed_logits, k=topk, dim=-1).indices
    hard_mask = torch.zeros_like(probs).scatter(-1, top_idx, 1.0)

    # 前向用硬 top-k，反向仍由 soft 概率提供梯度
    hard_gate = probs * hard_mask
    hard_gate = hard_gate / hard_gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    gate = hard_gate + probs - probs.detach()
    return gate


class AdaptiveHopTransformer(nn.Module):
    """
    跨跳 Hop-Token Transformer

    输入:
        feat_list: 长度为 K+1 的 list，每个元素形状 [N, d]
    输出:
        fused_feat: [N, r]
        hop_gate : [N, K+1]

    设计目标:
        1) 共享投影 d -> r，避免 concat 维度爆炸
        2) 节点级 sparse routing，默认 Top-2
        3) 轻量 Transformer 建模 hop-hop 交互
        4) 可选 teacher-guided routing；推理时 teacher=None，学生仍可独立工作
    """
    def __init__(
        self,
        input_dim,
        num_hops,
        bottleneck_dim=128,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ffn_dim=256,
        proj_dropout=0.1,
        attn_dropout=0.1,
        gate_topk=2,
        gate_tau=1.0,
        gate_mode='soft_topk_st',
        gate_hidden_dim=128,
        gate_dropout=0.1,
        label_dim=None,
        teacher_guided_routing=True,
        teacher_router_dim=64,
        teacher_guided_dropout=0.2,
        use_consistency_feature=True,
    ):
        super().__init__()

        if bottleneck_dim % transformer_heads != 0:
            raise ValueError(
                f'hop_bottleneck_dim={bottleneck_dim} 必须能被 hop_transformer_heads={transformer_heads} 整除。'
            )

        self.input_dim = input_dim
        self.num_hops = num_hops
        self.bottleneck_dim = bottleneck_dim
        self.gate_topk = gate_topk
        self.gate_tau = gate_tau
        self.gate_mode = gate_mode
        self.teacher_guided_routing = bool(teacher_guided_routing)
        self.teacher_guided_dropout = float(teacher_guided_dropout)
        self.use_consistency_feature = bool(use_consistency_feature)
        self.teacher_router_dim = int(teacher_router_dim) if self.teacher_guided_routing else 0

        # 每个 hop 使用同一个共享投影层: d -> r
        self.input_norm = nn.LayerNorm(input_dim)
        self.shared_proj = nn.Linear(input_dim, bottleneck_dim)
        self.proj_norm = nn.LayerNorm(bottleneck_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # hop token 的可学习位置/身份嵌入
        self.hop_embedding = nn.Parameter(torch.zeros(1, num_hops, bottleneck_dim))
        self.router_hop_bias = nn.Parameter(torch.zeros(num_hops))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.hop_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # router 输入 = 当前 hop token + 节点级全局上下文 + 可选 teacher 上下文 + 可选 consistency 标量
        router_in_dim = bottleneck_dim * 2

        if self.teacher_guided_routing:
            if label_dim is None:
                raise ValueError('teacher_guided_routing=True 时，需要提供 label_dim 以构造 teacher logits 投影。')
            self.teacher_logit_proj = nn.Sequential(
                nn.Linear(label_dim, teacher_router_dim),
                nn.GELU(),
                nn.LayerNorm(teacher_router_dim),
            )
            self.teacher_scale = nn.Parameter(torch.tensor(1.0))
            router_in_dim += teacher_router_dim
        else:
            self.teacher_logit_proj = None
            self.teacher_scale = None

        if self.use_consistency_feature:
            router_in_dim += 1

        self.router_mlp = nn.Sequential(
            nn.Linear(router_in_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 1)
        )

        self.out_norm = nn.LayerNorm(bottleneck_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout)
        )

        self.last_hop_gate = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.shared_proj.weight)
        nn.init.zeros_(self.shared_proj.bias)

        nn.init.normal_(self.hop_embedding, std=0.02)
        nn.init.zeros_(self.router_hop_bias)

        for module in self.router_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        if self.teacher_logit_proj is not None:
            for module in self.teacher_logit_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        for module in self.out_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _stack_feat_list(self, feat_list):
        if isinstance(feat_list, (list, tuple)):
            if len(feat_list) != self.num_hops:
                raise ValueError(
                    f'feat_list 长度 ({len(feat_list)}) 与 num_hops ({self.num_hops}) 不一致。'
                )
            return torch.stack(feat_list, dim=1)  # [N, H, d]

        if torch.is_tensor(feat_list):
            if feat_list.dim() != 3:
                raise ValueError(f'若直接传 Tensor，形状应为 [N, H, d]，得到 {tuple(feat_list.shape)}')
            if feat_list.size(1) != self.num_hops:
                raise ValueError(
                    f'输入 Tensor 的 hop 维度 ({feat_list.size(1)}) 与 num_hops ({self.num_hops}) 不一致。'
                )
            return feat_list

        raise TypeError('feat_list 必须是 list/tuple[Tensor] 或 Tensor[N, H, d]。')

    def _build_teacher_context(self, tokens, teacher_logits=None, teacher_consistency=None):
        """
        tokens: [N, H, r]
        返回 router 输入张量: [N, H, router_in_dim]
        """
        num_nodes, num_hops, _ = tokens.shape
        ctx_parts = []

        # 当前 hop token
        # 节点级全局上下文（所有 hop token 的均值）
        node_context = tokens.mean(dim=1, keepdim=True).expand(-1, num_hops, -1)
        ctx_parts.extend([tokens, node_context])

        # teacher-guided logits 上下文
        if self.teacher_guided_routing:
            if teacher_logits is None:
                teacher_context = tokens.new_zeros(num_nodes, self.teacher_router_dim)
            else:
                teacher_context = self.teacher_logit_proj(teacher_logits)
                # 训练时随机丢 teacher 路由上下文，减轻训练/推理失配
                if self.training and self.teacher_guided_dropout > 0:
                    keep_mask = (
                        torch.rand(num_nodes, 1, device=tokens.device) > self.teacher_guided_dropout
                    ).float()
                    teacher_context = teacher_context * keep_mask
                teacher_context = teacher_context * self.teacher_scale

            teacher_context = teacher_context.unsqueeze(1).expand(-1, num_hops, -1)
            ctx_parts.append(teacher_context)

        # teacher consistency / confidence 标量
        if self.use_consistency_feature:
            if teacher_consistency is None:
                consistency = tokens.new_zeros(num_nodes, 1)
            else:
                consistency = teacher_consistency
                if consistency.dim() == 1:
                    consistency = consistency.unsqueeze(-1)
                consistency = consistency.to(tokens.dtype)

            consistency = consistency.unsqueeze(1).expand(-1, num_hops, -1)
            ctx_parts.append(consistency)

        return torch.cat(ctx_parts, dim=-1)

    def _route_tokens(self, tokens, teacher_logits=None, teacher_consistency=None):
        """
        基于节点级上下文生成每个节点对每个 hop 的分数
        """
        router_input = self._build_teacher_context(
            tokens=tokens,
            teacher_logits=teacher_logits,
            teacher_consistency=teacher_consistency
        )

        router_logits = self.router_mlp(router_input).squeeze(-1)  # [N, H]
        router_logits = router_logits + self.router_hop_bias.unsqueeze(0)

        hop_gate = build_sparse_gate(
            router_logits,
            topk=self.gate_topk,
            tau=self.gate_tau,
            mode=self.gate_mode,
            training=self.training
        )
        return hop_gate, router_logits

    def forward(self, feat_list, teacher_logits=None, teacher_consistency=None, return_gate=True):
        """
        feat_list:
            list([N, d]) or Tensor[N, H, d]
        teacher_logits:
            [N, C]，训练阶段可传 teacher logits
        teacher_consistency:
            [N] 或 [N,1]，可传 teacher 一致性/可靠性分数
        """
        tokens = self._stack_feat_list(feat_list)      # [N, H, d]
        tokens = self.input_norm(tokens)
        tokens = self.shared_proj(tokens)              # [N, H, r]
        tokens = F.gelu(tokens)
        tokens = self.proj_dropout(tokens)
        tokens = self.proj_norm(tokens)

        # 先做 node-wise sparse routing
        hop_gate, _ = self._route_tokens(
            tokens=tokens,
            teacher_logits=teacher_logits,
            teacher_consistency=teacher_consistency
        )

        # hop token identity embedding
        tokens = tokens + self.hop_embedding[:, :tokens.size(1), :]

        # 稀疏门控后再做跨 hop attention
        gated_tokens = tokens * hop_gate.unsqueeze(-1)
        mixed_tokens = self.hop_transformer(gated_tokens)

        # 用同一个稀疏 gate 做池化，输出固定 r 维表示
        fused_feat = (mixed_tokens * hop_gate.unsqueeze(-1)).sum(dim=1)
        fused_feat = self.out_norm(fused_feat)
        fused_feat = self.out_proj(fused_feat)

        self.last_hop_gate = hop_gate.detach()

        if return_gate:
            return fused_feat, hop_gate
        return fused_feat


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        if param['distill_mode'] == 0:
            self.model_name = param['teacher']
        else:
            self.model_name = param['student']

        self.use_adaptive_hop = (
            param.get('distill_mode', 1) == 1 and
            "MLP" in self.model_name and
            int(param.get('use_adaptive_hop', 0)) == 1
        )

        # 为了兼容旧日志逻辑，保留这两个属性
        self.hop_weights = None
        self.last_hop_gate = None
        self.adaptive_hop_module = None

        if self.use_adaptive_hop:
            self.adaptive_hop_module = AdaptiveHopTransformer(
                input_dim=param.get('raw_feat_dim', param['feat_dim']),
                num_hops=param.get('feat_hops', 0) + 1,
                bottleneck_dim=param.get('hop_bottleneck_dim', 128),
                transformer_heads=param.get('hop_transformer_heads', 4),
                transformer_layers=param.get('hop_transformer_layers', 1),
                transformer_ffn_dim=param.get(
                    'hop_transformer_ffn_dim',
                    param.get('hop_bottleneck_dim', 128) * 2
                ),
                proj_dropout=param.get('hop_proj_dropout', 0.1),
                attn_dropout=param.get('hop_attn_dropout', 0.1),
                gate_topk=param.get('hop_gate_topk', 2),
                gate_tau=param.get('hop_gate_tau', 1.0),
                gate_mode=param.get('hop_gate_mode', 'soft_topk_st'),
                gate_hidden_dim=param.get('hop_gate_hidden_dim', 128),
                gate_dropout=param.get('hop_gate_dropout', 0.1),
                label_dim=param.get('label_dim', None),
                teacher_guided_routing=bool(param.get('teacher_guided_routing', 1)),
                teacher_router_dim=param.get('teacher_router_dim', 64),
                teacher_guided_dropout=param.get('teacher_guided_dropout', 0.2),
                use_consistency_feature=bool(param.get('teacher_use_consistency', 1)),
            )

        if "MLP" in self.model_name:
            self.encoder = MLP(
                num_layers=param['num_layers'],
                in_dim=param['feat_dim'],
                hidden_dim=param['hidden_dim'],
                out_dim=param['label_dim'],
                dropout_ratio=param.get('dropout_t', 0.5)
            ).to(device)
        elif "GCN" in self.model_name:
            self.encoder = GCN(
                num_layers=param['num_layers'],
                in_dim=param['feat_dim'],
                hidden_dim=param['hidden_dim'],
                out_dim=param['label_dim'],
                dropout_ratio=param['dropout_s'],
                activation=F.relu
            ).to(device)
        elif "GAT" in self.model_name:
            self.encoder = GAT(
                num_layers=param['num_layers'],
                in_dim=param['feat_dim'],
                hidden_dim=param['hidden_dim'],
                out_dim=param['label_dim'],
                dropout_ratio=param['dropout_s'],
                activation=F.relu,
                num_heads=param.get('num_heads', 8)
            ).to(device)
        elif "SAGE" in self.model_name:
            self.encoder = GraphSAGE(
                num_layers=param['num_layers'],
                in_dim=param['feat_dim'],
                hidden_dim=param['hidden_dim'],
                out_dim=param['label_dim'],
                dropout_ratio=param['dropout_s'],
                activation=F.relu
            ).to(device)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def forward(self, g, feats):
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(g, feats)

    def fuse_multi_hop_features(self, feat_list, teacher_logits=None, teacher_consistency=None, return_gate=True):
        if self.adaptive_hop_module is None:
            raise RuntimeError("Adaptive hop transformer is not enabled for the current model.")

        fused_feat, hop_gate = self.adaptive_hop_module(
            feat_list,
            teacher_logits=teacher_logits,
            teacher_consistency=teacher_consistency,
            return_gate=True
        )
        self.last_hop_gate = hop_gate.detach()

        if return_gate:
            return fused_feat, hop_gate
        return fused_feat
