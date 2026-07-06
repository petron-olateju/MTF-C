import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml


DATASET_LABELS = {
    'BNCI2014_001': 'BNCI 2014-001',
    'BNCI2014_002': 'BNCI 2014-002',
    'BNCI2014_004': 'BNCI 2014-004',
    'Cattan2019_PHMD': 'Cattan 2019',
    'Kalunga2016': 'Kalunga 2016',
    'Nakanishi2015': 'Nakanishi 2015',
    'Rodrigues2017': 'Rodrigues 2017',
}

_DEFAULT_SWEEP_LAMBDAS = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]


def _load_history(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    entries = []
    if isinstance(data, dict):
        for ts, entry in data.items():
            entry['timestamp'] = ts
            entries.append(entry)

    return entries


def _extract_dataset_name(dirname):
    m = re.search(r'dataset=([^|]+)', dirname)
    return m.group(1).strip() if m else dirname


def _params_match(entry_params, filter_params):
    if filter_params is None:
        return True
    for key, value in filter_params.items():
        if key in entry_params and entry_params[key] != value:
            return False
    return True


def plot_lambda_sweep(root_dir='experiments/classification/frequency_backbone',
                       save_path='plots/classification/frequency_backbone',
                       filename='lambda_sweep',
                       figsize=(10, 6.5),
                       training_params=None,
                       model_params=None,
                       sweep_lambdas=None):
    """Plot LOSO accuracy delta (%) vs db_conformer across a reconstruction-lambda sweep.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing dataset subdirectories.
    save_path : str or None
        Directory to save output files. Set to None to disable saving.
    filename : str
        Base name for output files (saved as {filename}.png and pdf/{filename}.pdf).
    figsize : tuple
        Figure dimensions (width, height) in inches.
    training_params : dict or None
        Only consider entries whose ``taining_params`` match all specified key-value
        pairs. Keys not present in the entry are ignored (default None = no filter).
    model_params : dict or None
        Only consider entries whose ``model_params`` match all specified key-value
        pairs. Keys not present in the entry are ignored (default None = no filter).
    sweep_lambdas : list of float or None
        Lambda values to include in the sweep. If None, uses
        [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0].
    """

    # ---- publication-style rcParams (scoped via context manager below) ----
    style = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'cmr10'],
        'mathtext.fontset': 'cm',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'axes.labelcolor': '#222222',
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        'savefig.dpi': 300,
        'figure.dpi': 120,
    }

    if sweep_lambdas is None:
        sweep_lambdas = list(_DEFAULT_SWEEP_LAMBDAS)
    x_tick_labels = ['0\n(off)'] + [str(lam) for lam in sweep_lambdas]

    # ---- data loading ----
    root = Path(root_dir)
    datasets = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        dataset_name = _extract_dataset_name(d.name)
        yaml_path = d / 'history.yaml'
        if not yaml_path.exists():
            continue

        entries = [
            e for e in _load_history(yaml_path)
            if _params_match(e.get('taining_params', {}), training_params)
            and _params_match(e.get('model_params', {}), model_params)
        ]

        db_conf_acc = None
        null_baseline_acc = None
        sweep_points = []

        for e in entries:
            model = e.get('model')
            if model == 'db_conformer':
                db_conf_acc = e.get('mean_acc')
            elif model == 'mtf_c':
                model_params_dict = e.get('model_params', {})
                sst = model_params_dict.get('sst_method')
                lam = model_params_dict.get('reconstruction_lambda')
                if sst is None and lam == 0.0:
                    null_baseline_acc = e.get('mean_acc')
                elif sst == 'frequency_backbone':
                    sweep_points.append((lam, e.get('mean_acc')))

        if db_conf_acc:
            null_delta = (
                (null_baseline_acc - db_conf_acc) / db_conf_acc * 100
                if null_baseline_acc is not None else None
            )
            sweep_deltas = [
                (lam, (acc - db_conf_acc) / db_conf_acc * 100)
                for lam, acc in sweep_points
            ]
        else:
            null_delta = None
            sweep_deltas = []

        if null_delta is not None:
            all_deltas = [(0.0, null_delta)] + sorted(sweep_deltas, key=lambda x: x[0])
        else:
            all_deltas = sorted(sweep_deltas, key=lambda x: x[0])

        datasets[dataset_name] = {'db_conformer_acc': db_conf_acc, 'all_deltas': all_deltas}

    # ---- plotting ----
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize)

        # Colorblind-safe, high-contrast palette (Wong 2011) + distinct markers
        palette = ['#0173B2', '#DE8F05', '#029E73', '#D55E00',
                   '#CC78BC', '#CA9161', '#949494', '#56B4E9']
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

        ax.axhline(y=0, color='#444444', linestyle='--', linewidth=1.3,
                   alpha=0.8, zorder=1, label='DBConformer baseline')

        for i, (dataset_name, data) in enumerate(datasets.items()):
            label = DATASET_LABELS.get(dataset_name, dataset_name)

            x_vals, y_vals = [], []
            for lam, delta in data['all_deltas']:
                if lam == 0.0:
                    x_vals.append(0)
                elif lam in sweep_lambdas:
                    x_vals.append(sweep_lambdas.index(lam) + 1)
                else:
                    continue
                y_vals.append(delta)

            if not x_vals:
                continue

            color = palette[i % len(palette)]
            marker = markers[i % len(markers)]

            ax.plot(x_vals, y_vals,
                    label=label,
                    color=color,
                    marker=marker,
                    markersize=6.5,
                    markeredgecolor='white',
                    markeredgewidth=0.7,
                    linewidth=2.0,
                    alpha=0.95,
                    zorder=3)

        # ---- axes cosmetics ----
        ax.set_xticks(range(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels, fontsize=11)
        ax.set_xlim(-0.3, len(x_tick_labels) - 0.3)

        ax.set_xlabel('Reconstruction $\\lambda$', fontsize=13, labelpad=8)
        ax.set_ylabel('LOSO Accuracy $\\Delta$ vs DBConformer (%)', fontsize=13, labelpad=8)
        ax.set_title('Effect of Reconstruction Weight on Classification Accuracy',
                     fontsize=14, fontweight='bold', pad=14)

        ax.grid(True, which='major', axis='y', alpha=0.3, linewidth=0.7, zorder=0)
        ax.grid(False, axis='x')
        ax.set_axisbelow(True)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_color('#333333')

        ax.tick_params(axis='both', which='major', labelsize=10.5, length=4)

        # Legend placed outside the axes so it never overlaps data/markers
        legend = ax.legend(fontsize=9.5, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                            frameon=True, framealpha=0.95, edgecolor='#cccccc',
                            borderpad=0.8, labelspacing=0.6)
        legend.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, 'pdf'), exist_ok=True)
            
            fig.savefig(os.path.join(save_path, f'{filename}.png'),
                        dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_path, 'pdf', f'{filename}.pdf'),
                        bbox_inches='tight')

    return fig, ax


def plot_comparison_bars(
    config_pairs,
    root_dir='experiments/classification/frequency_backbone',
    save_path='plots/classification/frequency_backbone',
    filename='comparison_bar',
    figsize=(12, 6),
    metric='mean_acc',
    error_metric='std_acc',
    datasets=None,
):
    """Grouped bar plot comparing accuracy across multiple (training_params, model_params) pairs.

    Parameters
    ----------
    config_pairs : dict of str -> tuple
        Maps a config label to ``(training_params, model_params, model)`` where
        ``model`` is optional (e.g. ``'db_conformer'`` or ``'mtf_c'``).
    root_dir : str or Path
        Directory containing dataset subdirectories.
    save_path : str or None
        Directory to save output files. Set to None to disable saving.
    filename : str
        Base name for output files (saved as {filename}.png and pdf/{filename}.pdf).
    figsize : tuple
        Figure dimensions (width, height) in inches.
    metric : str
        Key to extract from each entry (default ``'mean_acc'``).
    error_metric : str or None
        Key for error-bar values (default ``'std_acc'``). Set to None to omit.
    datasets : list of str or None
        List of dataset names to include. If None (default), all datasets are included.
    """
    style = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'cmr10'],
        'mathtext.fontset': 'cm',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'axes.labelcolor': '#222222',
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        'savefig.dpi': 300,
        'figure.dpi': 120,
    }

    root = Path(root_dir)
    config_labels = list(config_pairs.keys())
    n_configs = len(config_labels)

    # Parse each config item → (training_params, model_params, model_filter)
    config_filters = []
    for label in config_labels:
        item = config_pairs[label]
        if isinstance(item, dict):
            tp = item.get('training_params')
            mp = item.get('model_params')
            mf = item.get('model')
        elif isinstance(item, (list, tuple)):
            tp = item[0] if len(item) > 0 else None
            mp = item[1] if len(item) > 1 else None
            mf = item[2] if len(item) > 2 else None
        else:
            tp = mp = mf = None
        config_filters.append((tp, mp, mf))

    # Accumulate results: results[label][dataset_name] = (value, error)
    results = {label: {} for label in config_labels}

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        dataset_name = _extract_dataset_name(d.name)
        if datasets is not None and dataset_name not in datasets:
            continue
        yaml_path = d / 'history.yaml'
        if not yaml_path.exists():
            continue

        entries = _load_history(yaml_path)

        for label, (tp, mp, mf) in zip(config_labels, config_filters):
            matched = [
                e for e in entries
                if _params_match(e.get('taining_params', {}), tp)
                and _params_match(e.get('model_params', {}), mp)
                and (mf is None or e.get('model') == mf)
            ]
            if not matched:
                continue
            if len(matched) > 1:
                import warnings
                warnings.warn(
                    f"Multiple entries match config '{label}' in {dataset_name}; "
                    f"using the first one."
                )
            entry = matched[0]
            val = entry.get(metric)
            err = entry.get(error_metric) if error_metric else None
            if val is not None:
                results[label][dataset_name] = (val, err)

    # Collect all datasets that have data for at least one config
    all_datasets = sorted({
        dset
        for label in config_labels
        for dset in results[label]
    })
    if datasets is not None:
        all_datasets = sorted(set(all_datasets) & set(datasets))
    if not all_datasets:
        raise ValueError("No data found for any dataset / config pair.")

    # ---- plotting ----
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize)

        palette = ['#0173B2', '#DE8F05', '#029E73', '#D55E00',
                   '#CC78BC', '#CA9161', '#949494', '#56B4E9']

        n_groups = len(all_datasets)
        bar_width = 0.7 / n_configs
        x = np.arange(n_groups)

        for i, label in enumerate(config_labels):
            offsets = x + (i - (n_configs - 1) / 2) * bar_width
            vals = []
            errs = []
            for dset in all_datasets:
                v_e = results[label].get(dset)
                vals.append(v_e[0] if v_e else 0)
                errs.append(v_e[1] if v_e and v_e[1] is not None else 0)

            color = palette[i % len(palette)]
            bars = ax.bar(
                offsets, vals, bar_width,
                label=label,
                color=color,
                edgecolor='white',
                linewidth=0.6,
                alpha=0.92,
                zorder=3,
                yerr=errs if error_metric else None,
                capsize=3,
                error_kw={'linewidth': 1.2, 'ecolor': '#333333'},
            )

        # Labels and cosmetics
        dataset_labels = [
            DATASET_LABELS.get(d, d) for d in all_datasets
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=11, rotation=15, ha='right')

        metric_display = metric.replace('_', ' ').title()
        ax.set_ylabel(metric_display, fontsize=13, labelpad=8)
        ax.set_title('Comparison Across Configurations', fontsize=14,
                     fontweight='bold', pad=14)

        ax.grid(True, which='major', axis='y', alpha=0.3, linewidth=0.7, zorder=0)
        ax.grid(False, axis='x')
        ax.set_axisbelow(True)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_color('#333333')

        ax.tick_params(axis='both', which='major', labelsize=10.5, length=4)

        legend = ax.legend(fontsize=9.5, loc='upper left',
                           bbox_to_anchor=(1.02, 1.0),
                           frameon=True, framealpha=0.95, edgecolor='#cccccc',
                           borderpad=0.8, labelspacing=0.6)
        legend.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, 'pdf'), exist_ok=True)
            fig.savefig(os.path.join(save_path, f'{filename}.png'),
                        dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_path, 'pdf', f'{filename}.pdf'),
                        bbox_inches='tight')

    return fig, ax


def _parse_config_item(item):
    if isinstance(item, dict) and 'training_params' in item:
        return (item.get('training_params'), item.get('model_params'), item.get('model'))
    if isinstance(item, (list, tuple)):
        tp = item[0] if len(item) > 0 else None
        mp = item[1] if len(item) > 1 else None
        mf = item[2] if len(item) > 2 else None
        return (tp, mp, mf)
    return (None, None, None)


# def _find_entry(entries, tp, mp, mf):
#     matched = [
#         e for e in entries
#         if _params_match(e.get('taining_params', {}), tp)
#         and _params_match(e.get('model_params', {}), mp)
#         and (mf is None or e.get('model') == mf)
#     ]
#     if not matched:
#         return None, None
#     if len(matched) > 1:
#         import warnings
#         warnings.warn(
#             f"Multiple entries match config in {entries[0].get('dataset', 'unknown')}; "
#             f"using the first one."
#         )
#     entry = matched[0]
#     return entry.get('mean_acc'), entry.get('std_acc')

def _find_entry(entries, tp, mp, mf, metric='mean_acc', error_metric='std_acc'):
    matched = [
        e for e in entries
        if _params_match(e.get('taining_params', {}), tp)
        and _params_match(e.get('model_params', {}), mp)
        and (mf is None or e.get('model') == mf)
    ]
    if not matched:
        return None, None
    if len(matched) > 1:
        import warnings
        warnings.warn(
            f"Multiple entries match config in {entries[0].get('dataset', 'unknown')}; "
            f"using the first one."
        )
    entry = matched[0]
    return entry.get(metric), entry.get(error_metric)


def plot_paired_delta(
    config_pairs,
    reference_key='baseline',
    root_dir='experiments/classification/frequency_backbone',
    save_path='plots/classification/frequency_backbone',
    filename='paired_delta',
    figsize=(12, 6),
    metric='mean_acc',
    error_metric='std_acc',
    line=False,
    hline=None,
    datasets=None,
    percent=True,
):
    """Line plot of relative accuracy delta between paired configurations.

    Each entry in ``config_pairs`` is a dict with exactly two sub-configs
    (one identified by ``reference_key``, the other by an arbitrary second key).
    For every dataset the relative delta
    ``(acc_comparison - acc_reference) / acc_reference * 100``
    is computed and plotted as a connected line across datasets.

    Parameters
    ----------
    config_pairs : dict of str -> dict
        Maps a display label to a two-key dict of sub-configs. Example::

            {
                'Recon 0.5 vs baseline': {
                    'baseline': (training_params, model_params, 'mtf_c'),
                    'experiment': (training_params, model_params, 'mtf_c'),
                },
            }

        Each sub-config can be:

        - a tuple ``(training_params, model_params, model)``
        - a dict with keys ``'training_params'``, ``'model_params'``, ``'model'``

        ``model`` is optional (``None`` matches any model).
    reference_key : str
        Key name inside each entry that identifies the reference (denominator).
    root_dir : str or Path
        Directory containing dataset subdirectories.
    save_path : str or None
        Directory to save output files. Set to None to disable saving.
    filename : str
        Base name for output files (saved as {filename}.png and pdf/{filename}.pdf).
    figsize : tuple
        Figure dimensions (width, height) in inches.
    metric : str
        Metric key to extract from each entry (default ``'mean_acc'``).
    datasets : list of str or None
        List of dataset names to include. If None (default), all datasets are included.
    percent : bool
        If True (default), y-axis values are relative deltas expressed as a percentage
        ``(num - ref) / ref * 100``. If False, raw differences ``num - ref`` are shown.
    """
    style = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'cmr10'],
        'mathtext.fontset': 'cm',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'axes.labelcolor': '#222222',
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        'savefig.dpi': 300,
        'figure.dpi': 120,
    }

    root = Path(root_dir)
    labels = list(config_pairs.keys())
    n_labels = len(labels)

    # Parse each label → (denominator_subconfig, numerator_subconfig)
    parsed = []
    for label in labels:
        entry = config_pairs[label]
        if not isinstance(entry, dict):
            raise TypeError(
                f"Each config_pairs entry must be a dict; got {type(entry).__name__} "
                f"for label '{label}'."
            )
        keys = list(entry.keys())
        if reference_key not in keys:
            raise KeyError(
                f"reference_key='{reference_key}' not found in config_pairs['{label}']. "
                f"Available keys: {keys}"
            )
        if len(keys) != 2:
            raise ValueError(
                f"config_pairs['{label}'] has {len(keys)} keys; expected exactly 2 "
                f"(one reference, one comparison). Got: {keys}"
            )
        ref_key = reference_key
        num_key = [k for k in keys if k != ref_key][0]
        parsed.append((
            _parse_config_item(entry[ref_key]),
            _parse_config_item(entry[num_key]),
        ))

    # Accumulate deltas: deltas[label][dataset_name] = delta_value
    deltas = {label: {} for label in labels}

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        dataset_name = _extract_dataset_name(d.name)
        if datasets is not None and dataset_name not in datasets:
            continue
        yaml_path = d / 'history.yaml'
        if not yaml_path.exists():
            continue

        entries = _load_history(yaml_path)

        for label, ((ref_tp, ref_mp, ref_mf), (num_tp, num_mp, num_mf)) in zip(labels, parsed):
            ref_val, _ = _find_entry(entries, ref_tp, ref_mp, ref_mf, metric=metric, error_metric=error_metric)
            num_val, _ = _find_entry(entries, num_tp, num_mp, num_mf, metric=metric, error_metric=error_metric)
            if ref_val is not None and num_val is not None and ref_val != 0:
                delta = ((num_val - ref_val) / ref_val * 100) if percent else (num_val - ref_val)
                deltas[label][dataset_name] = delta

    all_datasets = sorted({
        dset
        for label in labels
        for dset in deltas[label]
    })
    if datasets is not None:
        all_datasets = sorted(set(all_datasets) & set(datasets))
    if not all_datasets:
        raise ValueError("No data found for any dataset / config pair.")

    # ---- plotting ----
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize)

        palette = ['#0173B2', '#DE8F05', '#029E73', '#D55E00',
                   '#CC78BC', '#CA9161', '#949494', '#56B4E9']
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

        x = np.arange(len(all_datasets))

        for i, label in enumerate(labels):
            vals = [deltas[label].get(dset, 0) for dset in all_datasets]
            color = palette[i % len(palette)]
            marker = markers[i % len(markers)]

            ax.plot(
                x, vals,
                label=label,
                color=color,
                marker=marker,
                markersize=8,
                markeredgecolor=color,
                markeredgewidth=1.5,
                markerfacecolor=color,
                linestyle='--' if line else 'None',
                linewidth=1.0 if line else 0,
                alpha=0.5 if line else None,
                zorder=3,
            )

        # Zero line
        ax.axhline(y=0, color='#444444', linestyle='--', linewidth=1.3,
                   alpha=0.8, zorder=2)

        # Additional horizontal reference lines
        if hline:
            for y_val, color in hline:
                ax.axhline(y=y_val, color=color, linestyle='--', linewidth=1.0,
                           alpha=0.7, zorder=2)

        # Labels and cosmetics
        dataset_labels = [
            DATASET_LABELS.get(d, d) for d in all_datasets
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=11, rotation=90, ha='center')

        suffix = ' (%)' if percent else ''
        ax.set_ylabel(
            f'{metric.replace("_", " ").title()} Relative Delta{suffix}',
            fontsize=13, labelpad=8,
        )
        ax.set_title(f'Paired Comparison – Relative Delta{suffix}', fontsize=14,
                      fontweight='bold', pad=14)

        ax.grid(True, which='major', axis='y', alpha=0.3, linewidth=0.7, zorder=0)
        ax.grid(False, axis='x')
        ax.set_axisbelow(True)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_color('#333333')

        ax.tick_params(axis='both', which='major', labelsize=10.5, length=4)

        legend = ax.legend(fontsize=9.5, loc='upper left',
                           bbox_to_anchor=(1.02, 1.0),
                           frameon=True, framealpha=0.95, edgecolor='#cccccc',
                           borderpad=0.8, labelspacing=0.6)
        legend.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, 'pdf'), exist_ok=True)
            fig.savefig(os.path.join(save_path, f'{filename}.png'),
                        dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_path, 'pdf', f'{filename}.pdf'),
                        bbox_inches='tight')

    return fig, ax