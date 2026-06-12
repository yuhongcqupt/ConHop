import argparse
import warnings
from pathlib import Path
import torch.optim as optim
import numpy as np
import os
import torch

from models import *
from dataloader import *
from train_and_eval import *

warnings.filterwarnings('ignore', category=Warning)


def main(param, g, feats, labels, indices):
    model = Model(param).to(device)
    teacher_out_dir = Path(param['teacher_out_dir'])

    if param['distill_mode'] == 0:
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=param['weight_decay'])
    elif param['distill_mode'] == 1:
        optimizer = optim.AdamW(model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    else:
        raise ValueError('distill_mode should be 0 or 1')

    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    evaluator = get_evaluator(param['dataset'])

    if param['distill_mode'] == 0:
        out, test_acc, test_val, test_best = train_teacher(
            param, model, g, feats, labels, indices, criterion_l, evaluator, optimizer
        )

        teacher_out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(teacher_out_dir / 'out.npz', out.detach().cpu())
        torch.save(model.state_dict(), teacher_out_dir / 'teacher_model.pt')
        return test_acc, test_val, test_best

    out_t = load_out_t(teacher_out_dir).to(device)

    teacher_model = None
    if int(param.get('use_consistency_weight', 0)) == 1:
        teacher_param = dict(param)
        teacher_param['distill_mode'] = 0
        teacher_param['feat_dim'] = feats.shape[1]
        teacher_model = Model(teacher_param).to(device)

        ckpt_path = teacher_out_dir / 'teacher_model.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f'Cannot find saved teacher checkpoint: {ckpt_path}. '
                f'Please train the teacher first with --distill_mode 0.'
            )
        teacher_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        teacher_model.eval()

    test_acc, test_val, test_best = train_student(
        param, model, g, feats, labels, out_t, indices,
        criterion_l, criterion_t, evaluator, optimizer,
        teacher_model=teacher_model
    )
    return test_acc, test_val, test_best


def resolve_config_path(args):
    if args.config_path is not None:
        return Path(args.config_path).expanduser()
    return Path(args.config_dir).expanduser() / f'{args.exp_setting}.conf.yaml'


def build_output_dir(param, run_id):
    return (
        Path(param['output_dir']).expanduser().resolve()
        / param['dataset']
        / param['teacher']
        / param['exp_setting']
        / str(run_id)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ConHop training script')
    parser.add_argument('--dataset', type=str, default='arxiv-year')
    parser.add_argument('--exp_setting', type=str, default='tran', choices=['tran', 'ind'], help='evaluation setting')
    parser.add_argument('--distill_mode', type=int, default=1, help='0: train teacher, 1: train student')
    parser.add_argument('--teacher', type=str, default='SAGE', help='teacher model')
    parser.add_argument('--student', type=str, default='MLP', help='student model')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Clean project-local paths.
    parser.add_argument('--data_dir', type=str, default='./data', help='directory for downloaded datasets')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='directory for teacher logits/checkpoints')
    parser.add_argument('--config_dir', type=str, default='./configs', help='directory containing tran.conf.yaml and ind.conf.yaml')
    parser.add_argument('--config_path', type=str, default=None, help='optional explicit YAML config path')

    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--split_id', type=int, default=0, help='split index for heterophilic datasets with built-in splits')
    parser.add_argument('--dropout_s', type=float, default=0.5)
    parser.add_argument('--dropout_t', type=float, default=0.1)

    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lamb', type=float, default=0.7)

    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--max_epoch', type=int, default=500)

    # CKD hyperparameters.
    parser.add_argument('--use_consistency_weight', type=int, default=1, help='1: enable consistency-aware KD weighting')
    parser.add_argument('--edge_drop_rate', type=float, default=0.2, help='edge dropout rate for the perturbed teacher view')
    parser.add_argument('--feat_drop_rate', type=float, default=0.1, help='feature dropout rate for the perturbed teacher view')
    parser.add_argument('--consistency_lambda', type=float, default=5.0, help='sharpness for consistency weighting')
    parser.add_argument('--weight_floor', type=float, default=0.2, help='minimum KD sample weight')

    # HSR hyperparameters.
    parser.add_argument('--feat_hops', type=int, default=4, help='number of propagation hops')
    parser.add_argument('--use_adaptive_hop', type=int, default=1, help='1: enable hop-token transformer fusion')
    parser.add_argument('--hop_bottleneck_dim', type=int, default=128, help='hop-token bottleneck dimension')
    parser.add_argument('--hop_transformer_heads', type=int, default=4, help='number of hop-token transformer heads')
    parser.add_argument('--hop_transformer_layers', type=int, default=1, help='number of hop-token transformer layers')
    parser.add_argument('--hop_transformer_ffn_dim', type=int, default=256, help='hidden dimension of the transformer FFN')
    parser.add_argument('--hop_proj_dropout', type=float, default=0.1, help='hop bottleneck projection dropout')
    parser.add_argument('--hop_attn_dropout', type=float, default=0.1, help='hop transformer attention dropout')
    parser.add_argument('--hop_gate_topk', type=int, default=5, help='number of selected hops for each node')
    parser.add_argument('--hop_gate_tau', type=float, default=1.0, help='softmax temperature for hop routing')
    parser.add_argument('--hop_gate_mode', type=str, default='soft', choices=['soft', 'soft_topk_st', 'gumbel_topk_st'])
    parser.add_argument('--hop_gate_hidden_dim', type=int, default=128, help='hidden dimension of the hop router MLP')
    parser.add_argument('--hop_gate_dropout', type=float, default=0.1, help='dropout rate of the hop router')
    parser.add_argument('--teacher_guided_routing', type=int, default=0, help='1: use teacher logits as auxiliary routing input')
    parser.add_argument('--teacher_router_dim', type=int, default=64, help='teacher-logit projection dimension')
    parser.add_argument('--teacher_guided_dropout', type=float, default=0.2, help='dropout for teacher-guided routing')
    parser.add_argument('--teacher_use_consistency', type=int, default=0, help='1: use teacher consistency weights as routing input')

    parser.add_argument('--feat_t_dim', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--split_rate', type=float, default=0.2)
    parser.add_argument('--num_runs', type=int, default=None, help='number of splits/seeds to run')
    parser.add_argument('--ablation_mode', type=int, default=0, help='0: full KD; 1: MLP only; 2: KD')

    args = parser.parse_args()

    param = dict(args.__dict__)
    config_path = resolve_config_path(args)
    if config_path.exists():
        config_param = get_train_config(config_path, args.student, args.dataset)
        param = dict(param, **config_param)
        print(f'Loaded config: {config_path}')
    elif args.config_path is not None:
        raise FileNotFoundError(f'Cannot find config file: {config_path}')
    else:
        print(f'Config file not found: {config_path}. Using command-line/default hyperparameters.')

    # Keep path arguments explicit and independent of YAML config values.
    param['data_dir'] = args.data_dir
    param['output_dir'] = args.output_dir
    param['config_dir'] = args.config_dir
    param['config_path'] = str(config_path)

    device = torch.device(param['device'])
    set_seed(param['seed'])

    g, labels = load_data(param['dataset'], param)
    g, labels = g.to(device), labels.to(device)

    results = []
    num_runs = args.num_runs if args.num_runs is not None else (5 if param['dataset'] == 'penn94' else 10)

    for i in range(num_runs):
        if param['dataset'] in homo_datasets:
            param['seed'] = i
            set_seed(param['seed'])
            run_id = param['seed']
        else:
            param['split_id'] = i
            run_id = param['split_id']

        teacher_out_dir = build_output_dir(param, run_id)
        param['teacher_out_dir'] = str(teacher_out_dir)

        idx_train, idx_val, idx_test = get_mask(g, param)

        if args.exp_setting == 'tran':
            indices = (idx_train, idx_val, idx_test)
        elif args.exp_setting == 'ind':
            indices = graph_split(idx_train, idx_val, idx_test, param['split_rate'], param['seed'])
        else:
            raise ValueError(f"Unsupported exp_setting: {args.exp_setting}")

        feats = g.ndata['feat'].to(device)
        raw_feat_dim = feats.shape[1]
        param['raw_feat_dim'] = raw_feat_dim

        if param['distill_mode'] == 0:
            param['feat_dim'] = raw_feat_dim
        else:
            if int(param.get('use_adaptive_hop', 0)) == 1:
                param['feat_dim'] = int(param.get('hop_bottleneck_dim', 128))
            else:
                param['feat_dim'] = raw_feat_dim * (param['feat_hops'] + 1)

        param['label_dim'] = labels.int().max().item() + 1

        test_acc, test_val, test_best = main(param, g, feats, labels, indices)
        results.append(round(test_val, 4))

    print(f'Student({args.student}): {results}')
    print(f'Mean: {np.mean(results):.4f}, STD: {np.std(results):.4f}')
    print('Configuration:', param)
