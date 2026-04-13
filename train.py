"""
DMG-PoE-CalSurv training script.
Train on a single cancer type with specified folds.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import SurvBoardLoader, BENCHMARK_CANCERS
from models.dmg_poe_calsurv import create_dmg_poe_model, DMGPoECalSurvLoss, CIndexMetric


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn,
    modality_indices: dict,
    device: str,
    max_time: float,
) -> dict:
    model.train()

    total_loss = 0.0
    total_nll = 0.0
    total_ibs = 0.0
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
        total_nll += losses['nll'].item()
        if 'ibs' in losses:
            total_ibs += losses['ibs'].item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'nll': total_nll / n_batches,
        'ibs': total_ibs / n_batches if total_ibs > 0 else 0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader,
    loss_fn,
    modality_indices: dict,
    device: str,
    max_time: float,
) -> dict:
    model.eval()

    all_risk_scores = []
    all_events = []
    all_times = []
    total_loss = 0.0
    n_batches = 0

    for batch in test_loader:
        features = batch['features'].to(device)
        events = batch['event'].to(device)
        times = batch['time'].to(device)

        outputs = model(features, modality_indices)
        losses = loss_fn(outputs, events, times, max_time)

        all_risk_scores.append(outputs['risk_score'].cpu())
        all_events.append(events.cpu())
        all_times.append(times.cpu())
        total_loss += losses['total'].item()
        n_batches += 1

    risk_scores = torch.cat(all_risk_scores)
    events = torch.cat(all_events)
    times = torch.cat(all_times)

    c_index = CIndexMetric.compute(risk_scores, events, times)

    return {
        'loss': total_loss / n_batches,
        'c_index': c_index,
    }


def train_fold(
    cancer: str,
    fold: int,
    config: dict,
    output_dir: Path,
    device: str = 'cuda',
) -> dict:
    print(f"\n{'='*50}")
    print(f"Training {cancer} Fold {fold}")
    print(f"{'='*50}")

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
    history = []

    for epoch in range(config['epochs']):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            data['modality_indices'], device, max_time
        )

        test_metrics = evaluate(
            model, test_loader, loss_fn,
            data['modality_indices'], device, max_time
        )

        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_nll': train_metrics['nll'],
            'train_ibs': train_metrics['ibs'],
            'test_loss': test_metrics['loss'],
            'test_c_index': test_metrics['c_index'],
        })

        if test_metrics['c_index'] > best_c_index:
            best_c_index = test_metrics['c_index']
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / f'{cancer}_fold{fold}_best.pt')

        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | "
                  f"Test C-index: {test_metrics['c_index']:.4f} (best: {best_c_index:.4f})")

    print(f"Best C-index: {best_c_index:.4f} at epoch {best_epoch}")

    return {
        'cancer': cancer,
        'fold': fold,
        'best_c_index': best_c_index,
        'best_epoch': best_epoch,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='Train DMG-PoE-CalSurv')
    parser.add_argument('--cancer', type=str, default='BRCA', help='Cancer type')
    parser.add_argument('--folds', type=int, nargs='+', default=[0], help='Folds to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--no_dmg', action='store_true', help='Disable DMG module')
    parser.add_argument('--no_poe', action='store_true', help='Disable PoE fusion')
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
        'use_calibration': True,
        'lambda_ibs': 0.1,
        'lambda_sc': 0.1,
        'use_incomplete': False,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        output_dir = Path('output') / f'{timestamp}_{args.cancer}'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"Config: {config}")
    print(f"Device: {args.device}")

    results = []
    for fold in args.folds:
        result = train_fold(args.cancer, fold, config, output_dir, args.device)
        results.append(result)

    c_indices = [r['best_c_index'] for r in results]
    print(f"\n{'='*50}")
    print(f"Results for {args.cancer}")
    print(f"{'='*50}")
    print(f"Folds: {args.folds}")
    print(f"C-indices: {c_indices}")
    print(f"Mean C-index: {np.mean(c_indices):.4f} +/- {np.std(c_indices):.4f}")

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == '__main__':
    main()
