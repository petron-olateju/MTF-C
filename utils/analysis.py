import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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


def _parse_entry(entry_text):
    result = {}

    m = re.search(r'^  model:\s*(.+)', entry_text, re.MULTILINE)
    if m:
        result['model'] = m.group(1).strip()

    m = re.search(r'^\s+n_repeats:\s*(\d+)', entry_text, re.MULTILINE)
    if m:
        result['n_repeats'] = int(m.group(1))

    m = re.search(r'^  mean_acc:\s*([\d.eE+\-]+)', entry_text, re.MULTILINE)
    if m:
        result['mean_acc'] = float(m.group(1))

    m = re.search(r'^  std_acc:\s*([\d.eE+\-]+)', entry_text, re.MULTILINE)
    if m:
        result['std_acc'] = float(m.group(1))

    m = re.search(r'^\s+sst_method:\s*(.+)', entry_text, re.MULTILINE)
    if m:
        val = m.group(1).strip()
        result['sst_method'] = None if val == 'null' else val

    m = re.search(r'^\s+reconstruction_lambda:\s*([\d.eE+\-]+)', entry_text, re.MULTILINE)
    if m:
        result['reconstruction_lambda'] = float(m.group(1))

    return result


def _load_history(path):
    text = path.read_text()
    timestamp_pat = r"^'(\d{4}-\d{2}-\d{2}T[\d:.]+)':"
    splits = re.split(timestamp_pat, text, flags=re.MULTILINE)

    entries = []
    for i in range(1, len(splits), 2):
        ts = splits[i]
        body = splits[i + 1] if i + 1 < len(splits) else ''
        entry = _parse_entry(body)
        entry['timestamp'] = ts
        entries.append(entry)

    return entries


def _extract_dataset_name(dirname):
    m = re.search(r'dataset=([^|]+)', dirname)
    return m.group(1).strip() if m else dirname


def plot_lambda_sweep(root_dir='experiments/classification/frequency_backbone',
                       save_path='plots/classification/frequency_backbone',
                       filename='lambda_sweep',
                       figsize=(10, 6.5),
                       n_repeats=5,
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
    n_repeats : int
        Only consider entries with this n_repeats value (default 5).
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

    # ---- data loading (unchanged logic) ----
    root = Path(root_dir)
    datasets = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        dataset_name = _extract_dataset_name(d.name)
        yaml_path = d / 'history.yaml'
        if not yaml_path.exists():
            continue

        entries = [e for e in _load_history(yaml_path) if e.get('n_repeats') == n_repeats]

        db_conf_acc = None
        null_baseline_acc = None
        sweep_points = []

        for e in entries:
            model = e.get('model')
            if model == 'db_conformer':
                db_conf_acc = e.get('mean_acc')
            elif model == 'mtf_c':
                sst = e.get('sst_method')
                lam = e.get('reconstruction_lambda')
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