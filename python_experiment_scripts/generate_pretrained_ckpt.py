import argparse
import re
from pathlib import Path
import yaml


# ── Helpers (mirroring utils/analysis.py) ──────────────────────────────────

def _load_history(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    entries = []
    if isinstance(data, dict):
        for ts, entry in data.items():
            entry['timestamp'] = ts
            entries.append(entry)
    return entries


def _params_match(entry_params, filter_params):
    if filter_params is None:
        return True
    for key, value in filter_params.items():
        if key in entry_params and entry_params[key] != value:
            return False
    return True


def _extract_dataset_name(dirname):
    m = re.search(r'dataset=([^|]+)', dirname)
    return m.group(1).strip() if m else dirname


def _parse_config_item(item):
    if isinstance(item, dict) and 'training_params' in item:
        return (item.get('training_params'), item.get('model_params'), item.get('model'))
    if isinstance(item, (list, tuple)):
        tp = item[0] if len(item) > 0 else None
        mp = item[1] if len(item) > 1 else None
        mf = item[2] if len(item) > 2 else None
        return (tp, mp, mf)
    return (None, None, None)


def _find_matching_entry(entries, tp, mp, mf):
    matched = [
        e for e in entries
        if _params_match(e.get('taining_params', {}), tp)
        and _params_match(e.get('model_params', {}), mp)
        and (mf is None or e.get('model') == mf)
    ]
    if not matched:
        return None
    return matched[-1]


# ── Model filter definitions ──────────────────────────────────────────────
# Edit these to match the exact config you want for each model.
# Follows the same convention as config_pairs in utils/analysis.py:
#   (training_params, model_params, model)
# where None means "match any".
#
# Example: pick db_conformer with n_repeats=5:
#   'db_conformer': ({'n_repeats': 5}, None, 'db_conformer')
#
# Example: pick mtf_c baseline (no reconstruction):
#   'mtf_c': (None, {'sst_method': None, 'reconstruction_lambda': 0.0}, 'mtf_c')
training_params = {'n_repeats': 1}
dbconformer_model_params = {
    'chn_depth': 2, 'chn_attn_flag': True, 
    'sst_method': 'stft', 'sst_decoder': 'st_projection+addition'
}
mtfc_model_params = {
    'chn_depth': 2, 'tem_depth': 2, 'spec_depth': 2, 
    'chn_attn_flag': True, 'spectrum_attn_flag': False, 'temporal_attn_flag': False,
    'reconstruction_lambda':0.001, 'sst_method': 'frequency_backbone', 'sst_decoder': 'sst_multi_projection+branch_addition'
    }
MODEL_FILTERS = {
    'db_conformer': (training_params, dbconformer_model_params, 'db_r_conformer'),
    'mtf_c': (
        training_params,
        mtfc_model_params,
        'mtf_r_c',
    ),
}

# Aliases that should reference the same entries as their target model.
MODEL_ALIASES = {
    'db_r_conformer': 'db_conformer',
    'mtf_r_c': 'mtf_c',
    'mtf_tr_c': 'mtf_r_c'
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate configs/pretrained_ckpt.yaml from history.yaml records.'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='experiments/classification/frequency_backbone',
        help='Root experiment directory containing dataset subdirectories.',
    )
    parser.add_argument(
        '--validation_strategy',
        type=str,
        default='loso',
        help='Validation strategy used in the experiment path naming.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='configs/pretrained_ckpt.yaml',
        help='Path for the generated YAML file.',
    )
    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f'Root directory not found: {root}')

    # Parse model filters into (training_params, model_params, model) tuples
    parsed_filters = {}
    for label, item in MODEL_FILTERS.items():
        parsed_filters[label] = _parse_config_item(item)

    # Collect timestamp paths per model per dataset
    checkpoints = {label: {} for label in parsed_filters}

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        dataset_name = _extract_dataset_name(d.name)
        yaml_path = d / 'history.yaml'
        if not yaml_path.exists():
            continue

        entries = _load_history(yaml_path)

        for label, (tp, mp, mf) in parsed_filters.items():
            entry = _find_matching_entry(entries, tp, mp, mf)
            if entry is None:
                print(
                    f'Warning: no matching entry for "{label}" in {dataset_name}'
                )
                continue

            timestamp = entry['timestamp']
            checkpoints[label][dataset_name] = f'{root}/{d.name}/{timestamp}|model:{mf}'

    # Build YAML data with shared objects (PyYAML will create anchors/aliases)
    yaml_data = {}
    for label in parsed_filters:
        yaml_data[label] = checkpoints[label] if checkpoints[label] else {}

    for alias, target in MODEL_ALIASES.items():
        if target in yaml_data:
            yaml_data[alias] = yaml_data[target]

    # Dump to YAML text, then rename auto-generated anchors
    yaml_text = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)

    # PyYAML assigns &id001, &id002, etc. in order of first appearance
    # We rename them to match our intended anchor names.
    anchor_idx = 1
    for alias, target in MODEL_ALIASES.items():
        auto_id = f'id{anchor_idx:03d}'
        yaml_text = yaml_text.replace(f'&{auto_id}', f'&{target}')
        # Handle both *idNNN and &idNNN (already replaced & above)
        anchor_idx += 1

    # Also replace the * references
    anchor_idx = 1
    for alias, target in MODEL_ALIASES.items():
        auto_id = f'id{anchor_idx:03d}'
        yaml_text = yaml_text.replace(f'*{auto_id}', f'*{target}')
        anchor_idx += 1

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(yaml_text)

    print(f'Generated {args.output}')
    for label in parsed_filters:
        n_datasets = len(checkpoints[label])
        print(f'  {label}: {n_datasets} datasets')


if __name__ == '__main__':
    main()
