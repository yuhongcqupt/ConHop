import copy
import time
import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np

from utils import *


def propagate_once(g, x):
    """
    对称归一化传播:
        X' = D^{-1/2} A D^{-1/2} X
    dataloader 中图已经加过 self-loop。
    """
    with g.local_scope():
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5).unsqueeze(1).to(x.device)

        g.ndata['h'] = x * norm
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        out = g.ndata['h'] * norm
    return out


def build_structure_features(g, feats, hops=2):
    """
    第一版拼接特征:
        [X, AX, A^2X, ...]
    """
    feat_list = [feats]
    h = feats
    for _ in range(hops):
        h = propagate_once(g, h)
        feat_list.append(h)
    return torch.cat(feat_list, dim=1)


def build_multi_hop_feature_list(g, feats, hops=2):
    """
    多阶特征列表:
        [X, AX, A^2X, ...]
    """
    feat_list = [feats]
    h = feats
    for _ in range(hops):
        h = propagate_once(g, h)
        feat_list.append(h)
    return feat_list


def prepare_student_features(model, g, feats, hops=2, use_adaptive_hop=0):
    """
    训练前准备学生输入特征。
    - adaptive_hop=0: 直接返回拼接后的静态特征
    - adaptive_hop=1: 返回多阶特征列表，供每个 epoch 重新融合
    """
    if int(use_adaptive_hop) == 1:
        return build_multi_hop_feature_list(g, feats, hops=hops)
    else:
        return build_structure_features(g, feats, hops=hops)


def materialize_student_features(
    model,
    prepared_feats,
    use_adaptive_hop=0,
    teacher_logits=None,
    teacher_consistency=None,
    return_gate=False,
):
    """
    每个 epoch 真正得到 student 输入。
    - adaptive_hop=0: prepared_feats 本身就是最终输入
    - adaptive_hop=1: 只返回固定维度 fused_feat，不再返回 concat 多跳特征

    参数:
        teacher_logits:
            训练阶段可传入 teacher logits，启用 teacher-guided routing。
            推理阶段传 None，学生独立运行。
        teacher_consistency:
            可选的一致性/置信度权重，形状 [N] 或 [N,1]。
        return_gate:
            若为 True，则返回 (fused_feat, hop_gate)。
    """
    if int(use_adaptive_hop) == 1:
        fused_feat, hop_gate = model.fuse_multi_hop_features(
            prepared_feats,
            teacher_logits=teacher_logits,
            teacher_consistency=teacher_consistency,
            return_gate=True
        )
        if return_gate:
            return fused_feat, hop_gate
        return fused_feat
    else:
        if return_gate:
            return prepared_feats, None
        return prepared_feats



def build_perturbed_graph(g, edge_drop_rate=0.2):
    """
    构造 teacher 的结构扰动视图：
    随机丢弃非自环边，再重新加 self-loop
    """
    src, dst = g.edges()
    device = src.device

    is_self_loop = (src == dst)
    keep_mask = torch.ones_like(src, dtype=torch.bool)

    nonself_idx = (~is_self_loop).nonzero(as_tuple=False).view(-1)
    if nonself_idx.numel() > 0 and edge_drop_rate > 0:
        rand_mask = torch.rand(nonself_idx.shape[0], device=device) > edge_drop_rate
        keep_mask[nonself_idx] = rand_mask

    new_src = src[keep_mask]
    new_dst = dst[keep_mask]

    pert_g = dgl.graph((new_src, new_dst), num_nodes=g.num_nodes(), device=device)
    pert_g = dgl.remove_self_loop(pert_g)
    pert_g = dgl.add_self_loop(pert_g)
    return pert_g


def js_divergence_from_logits(logits_p, logits_q, eps=1e-8):
    """
    节点级 JS divergence
    输入:
      logits_p: [N, C]
      logits_q: [N, C]
    返回:
      js: [N]
    """
    p = F.softmax(logits_p, dim=1).clamp(min=eps)
    q = F.softmax(logits_q, dim=1).clamp(min=eps)
    m = 0.5 * (p + q)

    js = 0.5 * ((p * (p.log() - m.log())).sum(dim=1) +
                (q * (q.log() - m.log())).sum(dim=1))
    return js


def compute_teacher_consistency_weights(
    teacher_model,
    g,
    feats,
    teacher_logits,
    edge_drop_rate=0.2,
    feat_drop_rate=0.1,
    consistency_lambda=5.0,
    weight_floor=0.2
):
    """
    用 teacher 原视图 vs 扰动视图的一致性，构造样本级蒸馏权重
    w_i = floor + (1-floor) * exp(-lambda * JS_i)
    """
    teacher_model.eval()
    with torch.no_grad():
        pert_g = build_perturbed_graph(g, edge_drop_rate=edge_drop_rate)
        pert_feats = F.dropout(feats, p=feat_drop_rate, training=True)

        pert_logits, _ = teacher_model(pert_g, pert_feats)

        js = js_divergence_from_logits(teacher_logits, pert_logits)
        weights = torch.exp(-consistency_lambda * js)

        # 设置下界，避免权重过小导致有效样本太少
        weights = weight_floor + (1.0 - weight_floor) * weights

    return weights


# =========================
# Teacher training / eval
# =========================

def train(model, data, feats, labels, criterion, optimizer, idx_train):
    model.train()

    logits, _ = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return logits, loss.item()


def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval):
    model.eval()

    with torch.no_grad():
        logits, _ = model(data, feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out[idx_eval], labels[idx_eval])
        acc = evaluator(out[idx_eval], labels[idx_eval])

    return logits, loss.item(), acc


def train_teacher(param, model, g, feats, labels, indices, criterion, evaluator, optimizer):
    device = param['device']
    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_g = g.subgraph(idx_obs).to(device)

    g.to(device)

    es = 0
    val_best = 0
    test_best = 0
    test_val = 0
    state = copy.deepcopy(model.state_dict())

    for epoch in range(1, param['max_epoch'] + 1):
        if param['exp_setting'] == 'tran':
            _, _ = train(model, g, feats, labels, criterion, optimizer, idx_train)
            _, train_loss, train_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_train)
            _, _, val_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_val)
            _, _, test_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_test)
        else:
            _, _ = train(model, obs_g, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
            _, train_loss, train_acc = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator, obs_idx_train)
            _, _, val_acc = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
            _, _, test_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_test_ind)

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1

        if es == 50:
            print("Early stopping!")
            break

        print(
            "\033[0;30;46m [{}] CLA: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
                epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_val, test_best
            )
        )

    model.load_state_dict(state)

    if param['exp_setting'] == 'tran':
        start_time = time.time()
        out, _, _ = evaluate(model, g, feats, labels, criterion, evaluator, idx_val)
        end_time = time.time()
    else:
        start_time = time.time()
        obs_out, _, _ = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
        out, _, _ = evaluate(model, g, feats, labels, criterion, evaluator, idx_test_ind)
        out[idx_obs] = obs_out
        end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time * 1000} ms")
    print("Test Val:{:.4f}".format(test_val))

    return out, test_acc, test_val, test_best


# =========================
# Student training / eval
# =========================

def evaluate_mini_batch(model, feats, labels, criterion, evaluator):
    model.eval()

    with torch.no_grad():
        logits, _ = model(None, feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, labels)
        acc = evaluator(out, labels)

    return loss.item(), acc


def train_mini_batch(model, feats, labels, out_t_all, idx_l, idx_kd, kd_weights, criterion_l, criterion_t, optimizer, param):
    model.train()

    logits, _ = model(None, feats)
    out = logits.log_softmax(dim=1)

    # 监督损失：只在有标签训练节点上计算
    loss_l = criterion_l(out[idx_l], labels[idx_l])

    # KD：在指定蒸馏节点集合上计算
    student_logp_T = (logits / param['tau']).log_softmax(dim=1)
    teacher_logp_T = (out_t_all / param['tau']).log_softmax(dim=1).to(student_logp_T.device)

    per_node_kd = F.kl_div(
        student_logp_T[idx_kd],
        teacher_logp_T[idx_kd],
        reduction='none',
        log_target=True
    ).sum(dim=1) * (param['tau'] * param['tau'])

    if kd_weights is not None:
        w = kd_weights[idx_kd].detach()
        w = w / (w.mean() + 1e-12)
        loss_t = (per_node_kd * w).mean()
    else:
        loss_t = per_node_kd.mean()

    if param['ablation_mode'] == 0:
        loss = loss_l * param['lamb'] + loss_t * (1 - param['lamb'])
    elif param['ablation_mode'] == 1:
        loss = loss_l
    elif param['ablation_mode'] == 2:
        loss = loss_l * param['lamb'] + loss_t * (1 - param['lamb'])
    else:
        loss = loss_l * param['lamb'] + loss_t * (1 - param['lamb'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_l.item(), loss_t.item()


def train_student(param, model, g, feats, labels, out_t_all, indices, criterion_l, criterion_t, evaluator, optimizer, teacher_model=None):
    hops = param.get('feat_hops', 2)
    use_adaptive_hop = int(param.get('use_adaptive_hop', 0))
    use_consistency_weight = int(param.get('use_consistency_weight', 0))
    teacher_guided_routing = int(param.get('teacher_guided_routing', 1))

    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
        idx_l = idx_train
        idx_kd = torch.arange(feats.shape[0], device=feats.device)

        prepared_feats = prepare_student_features(
            model=model,
            g=g,
            feats=feats,
            hops=hops,
            use_adaptive_hop=use_adaptive_hop
        )

        kd_weights = None
        if use_consistency_weight == 1 and teacher_model is not None:
            kd_weights = compute_teacher_consistency_weights(
                teacher_model=teacher_model,
                g=g,
                feats=feats,
                teacher_logits=out_t_all,
                edge_drop_rate=param.get('edge_drop_rate', 0.2),
                feat_drop_rate=param.get('feat_drop_rate', 0.1),
                consistency_lambda=param.get('consistency_lambda', 5.0),
                weight_floor=param.get('weight_floor', 0.2),
            )

        routing_teacher_logits_train = out_t_all if teacher_guided_routing == 1 else None
        routing_teacher_consistency_train = kd_weights if teacher_guided_routing == 1 else None

    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_idx_l = obs_idx_train

        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_out_t = out_t_all[idx_obs]

        idx_kd = torch.arange(obs_feats.shape[0], device=obs_feats.device)

        obs_g = g.subgraph(idx_obs).to(param['device'])

        obs_prepared_feats = prepare_student_features(
            model=model,
            g=obs_g,
            feats=obs_feats,
            hops=hops,
            use_adaptive_hop=use_adaptive_hop
        )

        full_prepared_feats = prepare_student_features(
            model=model,
            g=g,
            feats=feats,
            hops=hops,
            use_adaptive_hop=use_adaptive_hop
        )

        kd_weights = None
        if use_consistency_weight == 1 and teacher_model is not None:
            kd_weights = compute_teacher_consistency_weights(
                teacher_model=teacher_model,
                g=obs_g,
                feats=obs_feats,
                teacher_logits=obs_out_t,
                edge_drop_rate=param.get('edge_drop_rate', 0.2),
                feat_drop_rate=param.get('feat_drop_rate', 0.1),
                consistency_lambda=param.get('consistency_lambda', 5.0),
                weight_floor=param.get('weight_floor', 0.2),
            )

        routing_teacher_logits_train = obs_out_t if teacher_guided_routing == 1 else None
        routing_teacher_consistency_train = kd_weights if teacher_guided_routing == 1 else None

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    state = copy.deepcopy(model.state_dict())

    for epoch in range(1, param["max_epoch"] + 1):
        if param['exp_setting'] == 'tran':
            enhanced_feats_train, hop_gate = materialize_student_features(
                model=model,
                prepared_feats=prepared_feats,
                use_adaptive_hop=use_adaptive_hop,
                teacher_logits=routing_teacher_logits_train,
                teacher_consistency=routing_teacher_consistency_train,
                return_gate=True
            )
            enhanced_feats_eval = materialize_student_features(
                model=model,
                prepared_feats=prepared_feats,
                use_adaptive_hop=use_adaptive_hop,
                teacher_logits=None,
                teacher_consistency=None,
                return_gate=False
            )
        else:
            obs_enhanced_feats_train, hop_gate = materialize_student_features(
                model=model,
                prepared_feats=obs_prepared_feats,
                use_adaptive_hop=use_adaptive_hop,
                teacher_logits=routing_teacher_logits_train,
                teacher_consistency=routing_teacher_consistency_train,
                return_gate=True
            )
            obs_enhanced_feats_eval = materialize_student_features(
                model=model,
                prepared_feats=obs_prepared_feats,
                use_adaptive_hop=use_adaptive_hop,
                teacher_logits=None,
                teacher_consistency=None,
                return_gate=False
            )
            full_enhanced_feats_eval = materialize_student_features(
                model=model,
                prepared_feats=full_prepared_feats,
                use_adaptive_hop=use_adaptive_hop,
                teacher_logits=None,
                teacher_consistency=None,
                return_gate=False
            )

        if param['exp_setting'] == 'tran':
            loss_l, loss_t = train_mini_batch(
                model, enhanced_feats_train, labels, out_t_all, idx_l, idx_kd, kd_weights,
                criterion_l, criterion_t, optimizer, param
            )
            loss = loss_l + loss_t

            train_loss, train_acc = evaluate_mini_batch(model, enhanced_feats_eval[idx_train], labels[idx_train], criterion_l, evaluator)
            _, val_acc = evaluate_mini_batch(model, enhanced_feats_eval[idx_val], labels[idx_val], criterion_l, evaluator)
            _, test_acc = evaluate_mini_batch(model, enhanced_feats_eval[idx_test], labels[idx_test], criterion_l, evaluator)
        else:
            loss_l, loss_t = train_mini_batch(
                model, obs_enhanced_feats_train, obs_labels, obs_out_t, obs_idx_l, idx_kd, kd_weights,
                criterion_l, criterion_t, optimizer, param
            )
            loss = loss_l + loss_t

            train_loss, train_acc = evaluate_mini_batch(model, obs_enhanced_feats_eval[obs_idx_train], obs_labels[obs_idx_train], criterion_l, evaluator)
            _, val_acc = evaluate_mini_batch(model, obs_enhanced_feats_eval[obs_idx_val], obs_labels[obs_idx_val], criterion_l, evaluator)
            _, test_acc = evaluate_mini_batch(model, full_enhanced_feats_eval[idx_test_ind], labels[idx_test_ind], criterion_l, evaluator)

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1

        if es == 50:
            print("Early stopping!")
            break

        if use_adaptive_hop == 1 and hop_gate is not None:
            hop_alpha = hop_gate.detach().mean(dim=0).cpu().numpy()
            active_hops = (hop_gate.detach() > 0).float().sum(dim=1).float().mean().item()
            hop_alpha_str = ",".join([f"{x:.3f}" for x in hop_alpha])
            hop_info = f" | H=[{hop_alpha_str}] | AH={active_hops:.1f}"
        else:
            hop_info = ""

        if kd_weights is not None:
            weight_info = (
                f" | KW={kd_weights.mean().item():.3f}/"
                f"{kd_weights.min().item():.3f}/"
                f"{kd_weights.max().item():.3f}"
            )
        else:
            weight_info = ""

        log_msg = (
            f"[{epoch:03d}] "
            f"L={loss:.4f}(CE={loss_l:.4f},KD={loss_t:.4f}) "
            f"| Acc={train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f} "
            f"| Best={val_best:.4f}/{test_val:.4f}/{test_best:.4f}"
            f"{hop_info}{weight_info}"
        )

        # print(f"\033[0;1;41m{log_msg}\033[0m")

    model.load_state_dict(state)
    model.eval()

    if param['exp_setting'] == 'tran':
        enhanced_feats = materialize_student_features(
            model=model,
            prepared_feats=prepared_feats,
            use_adaptive_hop=use_adaptive_hop,
            teacher_logits=None,
            teacher_consistency=None,
            return_gate=False
        )
        start_time = time.time()
        _, _ = evaluate_mini_batch(model, enhanced_feats[idx_val], labels[idx_val], criterion_l, evaluator)
        end_time = time.time()
    else:
        obs_enhanced_feats = materialize_student_features(
            model=model,
            prepared_feats=obs_prepared_feats,
            use_adaptive_hop=use_adaptive_hop,
            teacher_logits=None,
            teacher_consistency=None,
            return_gate=False
        )
        start_time = time.time()
        _, _ = evaluate_mini_batch(model, obs_enhanced_feats, obs_labels, criterion_l, evaluator)
        end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time * 1000} ms")

    return test_acc, test_val, test_best

