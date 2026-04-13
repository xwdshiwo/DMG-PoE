"""
SurvBoard data loader for DMG-PoE.

Loads preprocessed multi-omics cancer survival data from the SurvBoard benchmark.
See: https://github.com/BoevaLab/SurvBoard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# Modality prefix mapping (SurvBoard convention)
MODALITY_PREFIXES = {
    'clinical': 'clinical_',
    'cnv': 'cnv_',
    'gex': 'gex_',
    'mirna': 'mirna_',
    'meth': 'meth_',
    'rppa': 'rppa_',
    'mut': 'mut_',
}

# Default data root (relative to project root)
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data" / "SurvBoard"

# 10 TCGA cancer types used in the paper
BENCHMARK_CANCERS = [
    'BLCA', 'BRCA', 'CESC', 'ESCA', 'GBM',
    'LGG', 'LUAD', 'PAAD', 'SARC', 'SKCM',
]


class SurvBoardDataset(Dataset):
    """SurvBoard survival analysis dataset."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,  # (N, 2) -> [event, time]
        modality_masks: Optional[np.ndarray] = None,  # (N, num_modalities)
        modality_indices: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.modality_masks = modality_masks
        self.modality_indices = modality_indices

        if modality_masks is not None:
            self.modality_masks = torch.FloatTensor(modality_masks)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'event': self.labels[idx, 0],
            'time': self.labels[idx, 1],
        }
        if self.modality_masks is not None:
            item['modality_mask'] = self.modality_masks[idx]
        return item


class SurvBoardLoader:
    """SurvBoard data loading manager."""

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self._cache = {}
        self._label_encoders = {}

    def _encode_categorical_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical columns to numeric."""
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = df[col].fillna('__MISSING__')
                if col not in self._label_encoders:
                    self._label_encoders[col] = LabelEncoder()
                    self._label_encoders[col].fit(df[col].astype(str))
                known_classes = set(self._label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if str(x) in known_classes else '__MISSING__')
                if '__MISSING__' not in known_classes:
                    self._label_encoders[col].classes_ = np.append(self._label_encoders[col].classes_, '__MISSING__')
                df[col] = self._label_encoders[col].transform(df[col].astype(str))
        return df

    def get_modality_columns(self, df: pd.DataFrame, modality: str) -> List[str]:
        """Get column names for a given modality."""
        prefix = MODALITY_PREFIXES.get(modality, f"{modality}_")
        return [c for c in df.columns if c.startswith(prefix)]

    def get_all_modality_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get column names for all modalities."""
        result = {}
        for mod_name, prefix in MODALITY_PREFIXES.items():
            cols = [c for c in df.columns if c.startswith(prefix)]
            if cols:
                result[mod_name] = cols
        return result

    def load_cancer_data(
        self,
        cancer: str,
        source: str = "TCGA",
        use_incomplete: bool = False,
    ) -> Dict:
        """
        Load data for a single cancer type.

        Args:
            cancer: Cancer type name, e.g. 'BRCA', 'LUAD'
            source: Data source, e.g. 'TCGA', 'ICGC'
            use_incomplete: Whether to load incomplete modality samples

        Returns:
            Dict with features, labels, splits, modality_info
        """
        cache_key = f"{source}_{cancer}_{use_incomplete}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        base_path = self.data_root / source
        splits_path = self.data_root / "splits" / source

        complete_file = base_path / f"{cancer}_data_complete_modalities_preprocessed.csv"
        df_complete = pd.read_csv(complete_file)

        labels_complete = df_complete[['OS', 'OS_days']].values

        modality_columns = self.get_all_modality_columns(df_complete)

        all_feature_cols = []
        for cols in modality_columns.values():
            all_feature_cols.extend(cols)
        df_complete = self._encode_categorical_columns(df_complete, all_feature_cols)

        feature_cols = []
        modality_indices = {}
        start_idx = 0

        for mod_name in ['clinical', 'cnv', 'gex', 'mirna', 'meth', 'rppa', 'mut']:
            if mod_name in modality_columns:
                cols = modality_columns[mod_name]
                feature_cols.extend(cols)
                modality_indices[mod_name] = (start_idx, start_idx + len(cols))
                start_idx += len(cols)

        features_complete = df_complete[feature_cols].values.astype(np.float32)
        features_complete = np.nan_to_num(features_complete, nan=0.0)

        train_splits_file = splits_path / f"{cancer}_train_splits.csv"
        test_splits_file = splits_path / f"{cancer}_test_splits.csv"

        train_splits = pd.read_csv(train_splits_file)
        test_splits = pd.read_csv(test_splits_file)

        result = {
            'features_complete': features_complete,
            'labels_complete': labels_complete,
            'train_splits': train_splits,
            'test_splits': test_splits,
            'modality_indices': modality_indices,
            'modality_columns': modality_columns,
            'feature_columns': feature_cols,
            'n_complete': len(features_complete),
        }

        if use_incomplete:
            incomplete_file = base_path / f"{cancer}_data_incomplete_modalities_preprocessed.csv"
            if incomplete_file.exists():
                df_incomplete = pd.read_csv(incomplete_file)
                labels_incomplete = df_incomplete[['OS', 'OS_days']].values

                df_incomplete = self._encode_categorical_columns(df_incomplete, feature_cols)

                features_incomplete = df_incomplete[feature_cols].values.astype(np.float32)

                modality_masks = self._compute_modality_masks(
                    df_incomplete, feature_cols, modality_indices
                )

                features_incomplete = np.nan_to_num(features_incomplete, nan=0.0)

                result['features_incomplete'] = features_incomplete
                result['labels_incomplete'] = labels_incomplete
                result['modality_masks_incomplete'] = modality_masks
                result['n_incomplete'] = len(features_incomplete)

        self._cache[cache_key] = result
        return result

    def _compute_modality_masks(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        modality_indices: Dict[str, Tuple[int, int]]
    ) -> np.ndarray:
        """Compute per-sample modality availability mask."""
        n_samples = len(df)
        n_modalities = len(modality_indices)
        masks = np.ones((n_samples, n_modalities), dtype=np.float32)

        for i, (mod_name, (start, end)) in enumerate(modality_indices.items()):
            mod_cols = feature_cols[start:end]
            mod_data = df[mod_cols].values
            all_nan = np.all(np.isnan(mod_data), axis=1)
            masks[all_nan, i] = 0.0

        return masks

    def get_fold_data(
        self,
        cancer: str,
        fold: int,
        source: str = "TCGA",
        use_incomplete_for_train: bool = False,
    ) -> Tuple[SurvBoardDataset, SurvBoardDataset]:
        """
        Get train and test datasets for a given fold.

        Args:
            cancer: Cancer type name
            fold: Fold number (0-24 for 5-fold × 5-repeat)
            source: Data source
            use_incomplete_for_train: Whether to augment training set with incomplete samples

        Returns:
            (train_dataset, test_dataset)
        """
        data = self.load_cancer_data(cancer, source, use_incomplete=use_incomplete_for_train)

        train_indices = data['train_splits'].iloc[fold].dropna().astype(int).values
        test_indices = data['test_splits'].iloc[fold].dropna().astype(int).values

        train_features = data['features_complete'][train_indices]
        train_labels = data['labels_complete'][train_indices]

        if use_incomplete_for_train and 'features_incomplete' in data:
            train_features = np.vstack([train_features, data['features_incomplete']])
            train_labels = np.vstack([train_labels, data['labels_incomplete']])

            complete_masks = np.ones((len(train_indices), len(data['modality_indices'])))
            all_masks = np.vstack([complete_masks, data['modality_masks_incomplete']])
        else:
            all_masks = None

        train_dataset = SurvBoardDataset(
            features=train_features,
            labels=train_labels,
            modality_masks=all_masks,
            modality_indices=data['modality_indices'],
        )

        test_features = data['features_complete'][test_indices]
        test_labels = data['labels_complete'][test_indices]

        test_dataset = SurvBoardDataset(
            features=test_features,
            labels=test_labels,
            modality_indices=data['modality_indices'],
        )

        return train_dataset, test_dataset

    def get_dataloader(
        self,
        dataset: SurvBoardDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create DataLoader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )


def get_available_cancers(data_root: Optional[Path] = None, source: str = "TCGA") -> List[str]:
    """Get list of available cancer types."""
    root = Path(data_root) if data_root else DATA_ROOT
    source_path = root / source
    cancers = []
    for f in source_path.glob("*_data_complete_modalities_preprocessed.csv"):
        cancer = f.stem.replace("_data_complete_modalities_preprocessed", "")
        cancers.append(cancer)
    return sorted(cancers)


if __name__ == "__main__":
    print("Testing SurvBoardLoader...")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"DATA_ROOT exists: {DATA_ROOT.exists()}")

    if not DATA_ROOT.exists():
        print("\nData not found. Please download SurvBoard data first.")
        print("See README.md for instructions.")
    else:
        loader = SurvBoardLoader()

        print("\n--- Loading BRCA ---")
        data = loader.load_cancer_data("BRCA", "TCGA")
        print(f"Complete samples: {data['n_complete']}")
        print(f"Feature dim: {data['features_complete'].shape[1]}")
        print(f"Modalities: {list(data['modality_indices'].keys())}")
        for mod, (start, end) in data['modality_indices'].items():
            print(f"  {mod}: {end - start} features")

        print("\n--- Testing Fold 0 ---")
        train_ds, test_ds = loader.get_fold_data("BRCA", fold=0)
        print(f"Train samples: {len(train_ds)}")
        print(f"Test samples: {len(test_ds)}")

        print("\nAll tests passed!")
