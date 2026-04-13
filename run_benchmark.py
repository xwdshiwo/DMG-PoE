"""
DMG-PoE-CalSurv full benchmark script.
Runs multi-cancer, multi-fold experiments with sample-level prediction output.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_loader import SurvBoardLoader, BENCHMARK_CANCERS
from models.dmg_poe_calsurv import create_dmg_poe_model, DMGPoECalSurvLoss, CIndexMetric


def train_one_epoch(model, train_loader, optimizer, loss_fn, modality_indices, device, max_time):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        features = batch['features'].to(device)
        events = batch['event'].to(device)
        times = batch['time'].to(device)

        modality_mask = batch.get('modality_mask')
        if modality_mask is not None:
            modality_mask = modality_mask.to(device)

        optimizer.zero_grad()
        outputs = model(features, modality_indices, modality_mask)
        losses = loss_fn(outputs, events, times, max_time)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        n_batches += 1

    return {'loss': total_loss / n_batches}


@torch.no_grad()
def evaluate_with_predictions(model, test_loader, loss_fn, modality_indices, device, max_time, num_time_bins):
    model.eval()

    all_risk_scores = []
    all_survival_probs = []
    all_events = []
    all_times = []
    all_sample_ids = []
    total_loss = 0.0
    n_batches = 0

    for batch in test_loader:
        features = batch['features'].to(device)
        events = batch['event'].to(device)
        times = batch['time'].to(device)
        sample_ids = batch.get('sample_id', list(range(len(events))))

        outputs = model(features, modality_indices)
        losses = loss_fn(outputs, events, times, max_time)

        all_risk_scores.append(outputs['risk_score'].cpu().numpy())
        all_events.append(events.cpu().numpy())
        all_times.append(times.cpu().numpy())

        if 'survival_probs' in outputs:
            all_survival_probs.append(outputs['survival_probs'].cpu().numpy())

        if isinstance(sample_ids, torch.Tensor):
            all_sample_ids.extend(sample_ids.cpu().numpy().tolist())
        else:
            all_sample_ids.extend(sample_ids)

        total_loss += losses['total'].item()
        n_batches += 1

    risk_scores = np.concatenate(all_risk_scores)
    events = np.concatenate(all_events)
    times = np.concatenate(all_times)

    c_index = CIndexMetric.compute(
        torch.tensor(risk_scores),
        torch.tensor(events),
        torch.tensor(times)
    )

    predictions_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'true_time': times,
        'true_event': events.astype(int),
        'risk_score': risk_scores,
    })

    if all_survival_probs:
        survival_probs = np.concatenate(all_survival_probs)
        time_bins = np.linspace(0, max_time, num_time_bins + 1)[1:]
        for year, days in [(1, 365), (3, 1095), (5, 1825)]:
            if days <= max_time:
                idx = np.searchsorted(time_bins, days)
                idx = min(idx, survival_probs.shape[1] - 1)
                predictions_df[f'surv_prob_{year}y'] = survival_probs[:, idx]

    return {
        'loss': total_loss / n_batches,
        'c_index': c_index,
        'predictions': predictions_df,
    }


def train_single_fold(cancer, fold, config, output_dir, device='cuda'):
    print(f"  Fold {fold}...", end=" ", flush=True)

    loader = SurvBoardLoader()
    data = loader.load_cancer_data(cancer, "TCGA", use_incomplete=config.get('use_incomplete', False))

    train_ds, test_ds = loader.get_fold_data(
        cancer, fold,
        use_incomplete_for_train=config.get('use_incomplete', False)
    )

    train_loader = loader.get_dataloader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_loader = loader.get_dataloader(test_ds, batch_size=config['batch_size'], shuffle=False)

    max_time = data['labels_complete'][:, 1].max()

    model = create_dmg_poe_model(
        data,
        hidden_dim=config['hidden_dim'],
        embed_dim=config['embed_dim'],
        num_time_bins=config['num_time_bins'],
        use_dmg=config.get('use_dmg', True),
        use_poe=config.get('use_poe', True),
        dmg_layers=config.get('dmg_layers', 2),
        dmg_heads=config.get('dmg_heads', 4),
        dropout=config.get('dropout', 0.3),
        fusion_type=config.get('fusion_type', 'hybrid_poe'),
        imputation_type=config.get('imputation_type', 'dmg'),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01)

    loss_fn = DMGPoECalSurvLoss(
        num_time_bins=config['num_time_bins'],
        lambda_ibs=config.get('lambda_ibs', 0.1),
        lambda_sc=config.get('lambda_sc', 0.1),
        use_calibration=config.get('use_calibration', True),
    )

    best_c_index = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(config['epochs']):
        train_one_epoch(model, train_loader, optimizer, loss_fn,
                       data['modality_indices'], device, max_time)

        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            eval_result = evaluate_with_predictions(
                model, test_loader, loss_fn,
                data['modality_indices'], device, max_time, config['num_time_bins']
            )

            if eval_result['c_index'] > best_c_index:
                best_c_index = eval_result['c_index']
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    if best_model_state:
        model.load_state_dict(best_model_state)

    final_result = evaluate_with_predictions(
        model, test_loader, loss_fn,
        data['modality_indices'], device, max_time, config['num_time_bins']
    )

    pred_path = output_dir / f'predictions_{cancer}_fold{fold}.csv'
    final_result['predictions'].to_csv(pred_path, index=False)

    print(f"C-index: {final_result['c_index']:.4f}")

    return {
        'cancer': cancer,
        'fold': fold,
        'c_index': final_result['c_index'],
        'best_epoch': best_epoch,
    }


def run_cancer_benchmark(cancer, config, output_dir, device='cuda', n_folds=25):
    print(f"\n{'='*60}")
    print(f"  {cancer} Benchmark ({n_folds} folds)")
    print(f"{'='*60}")

    results = []
    for fold in range(n_folds):
        try:
            result = train_single_fold(cancer, fold, config, output_dir, device)
            results.append(result)
        except Exception as e:
            print(f"  Fold {fold} FAILED: {e}")
            continue

    if not results:
        return None

    c_indices = [r['c_index'] for r in results]
    summary = {
        'cancer': cancer,
        'n_folds': len(results),
        'c_index_mean': float(np.mean(c_indices)),
        'c_index_std': float(np.std(c_indices)),
        'c_index_min': float(np.min(c_indices)),
        'c_index_max': float(np.max(c_indices)),
        'fold_results': results,
    }

    print(f"\n  {cancer} Summary: C-index = {summary['c_index_mean']:.4f} +/- {summary['c_index_std']:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run DMG-PoE-CalSurv Benchmark')
    parser.add_argument('--cancers', type=str, nargs='+', default=BENCHMARK_CANCERS,
                       help='Cancer types to benchmark')
    parser.add_argument('--folds', type=int, default=25, help='Number of folds (default: 25)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--no_dmg', action='store_true', help='Disable DMG module')
    parser.add_argument('--no_poe', action='store_true', help='Disable PoE fusion')
    parser.add_argument('--no_calibration', action='store_true', help='Disable calibration losses')
    parser.add_argument('--lambda_ibs', type=float, default=0.1, help='IBS loss weight')
    parser.add_argument('--lambda_sc', type=float, default=0.1, help='Subset consistency loss weight')
    parser.add_argument('--fusion_type', type=str, default='hybrid_poe',
                       choices=['hybrid_poe', 'poe', 'moe', 'average', 'concat'],
                       help='Fusion method type')
    parser.add_argument('--imputation_type', type=str, default='dmg',
                       choices=['dmg', 'zero', 'mean'],
                       help='Missing modality imputation strategy')
    parser.add_argument('--dmg_layers', type=int, default=2, help='Number of DMG layers')
    parser.add_argument('--dmg_heads', type=int, default=4, help='Number of DMG attention heads')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--use_incomplete', action='store_true', help='Use incomplete samples for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default=None, help='Custom output directory')
    args = parser.parse_args()

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'hidden_dim': args.hidden_dim,
        'embed_dim': args.embed_dim,
        'num_time_bins': 20,
        'use_dmg': not args.no_dmg,
        'use_poe': not args.no_poe,
        'use_calibration': not args.no_calibration,
        'lambda_ibs': args.lambda_ibs,
        'lambda_sc': args.lambda_sc,
        'fusion_type': args.fusion_type,
        'imputation_type': args.imputation_type,
        'dmg_layers': args.dmg_layers,
        'dmg_heads': args.dmg_heads,
        'dropout': args.dropout,
        'use_incomplete': args.use_incomplete,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        setting = 'Missing' if args.use_incomplete else 'Standard'
        output_dir = Path('output') / f'{timestamp}_Benchmark_{setting}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DMG-PoE-CalSurv Benchmark")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Cancers: {args.cancers}")
    print(f"Folds: {args.folds}")
    print(f"Config: {config}")
    print(f"Device: {args.device}")

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    all_results = []
    for cancer in args.cancers:
        try:
            result = run_cancer_benchmark(cancer, config, output_dir, args.device, args.folds)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n{cancer} FAILED: {e}")
            continue

    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'cancer': r['cancer'],
            'method': 'DMG-PoE-CalSurv',
            'c_index_mean': r['c_index_mean'],
            'c_index_std': r['c_index_std'],
            'n_folds': r['n_folds'],
            'setting': 'Missing' if config['use_incomplete'] else 'Standard',
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    if summary_rows:
        avg_c_index = np.mean([r['c_index_mean'] for r in all_results])
        print(f"\nOverall Average C-index: {avg_c_index:.4f}")

    print("\n" + "=" * 60)
    print("  Benchmark Complete!")
    print("=" * 60)

    return all_results


if __name__ == '__main__':
    main()
