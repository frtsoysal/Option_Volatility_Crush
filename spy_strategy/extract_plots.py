"""Extract embedded PNG outputs from executed notebooks into spy_strategy/data/.

Walks notebook cells in order; the Nth PNG encountered gets the Nth name from
the per-notebook FIGURE_NAMES mapping. Extras fall back to figure_N.png.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

# Per-notebook filename schedules. Keyed by notebook stem.
FIGURE_NAMES = {
    '00_data_sanity': [
        'iv_vs_vix_overlay.png',
        'iv_vs_vix_scatter.png',
    ],
    '03_ml_pipeline': [
        'confusion_matrix_test_all_models.png',
        'reliability_diagram_test.png',
        'predicted_proba_distribution.png',
        'shap_summary_lgbm.png',
        'feature_importance_lgbm.png',
        'predicted_proba_timeseries.png',
        'equity_curves_test.png',
    ],
    '04_backtest': [
        'bt_equity_curves_log.png',
        'bt_deployed_margin.png',
        'bt_underwater_drawdown.png',
        'bt_per_trade_pnl_hist.png',
        'bt_monthly_heatmap_winner.png',
        'bt_capital_efficiency.png',
    ],
}


def extract(nb_path: Path, out_dir: Path) -> list[Path]:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = FIGURE_NAMES.get(nb_path.stem, [])
    written: list[Path] = []
    fig_idx = 0

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        for out in cell.get('outputs', []):
            data = out.get('data') or {}
            png_b64 = data.get('image/png')
            if not png_b64:
                continue
            if isinstance(png_b64, list):
                png_b64 = ''.join(png_b64)
            payload = base64.b64decode(png_b64)

            if fig_idx < len(schedule):
                name = schedule[fig_idx]
            else:
                name = f'{nb_path.stem}_figure_{fig_idx + 1}.png'
            target = out_dir / name
            target.write_bytes(payload)
            written.append(target)
            fig_idx += 1

    return written


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    out_dir = here / 'data'

    notebooks = sys.argv[1:] or ['00_data_sanity.ipynb', '03_ml_pipeline.ipynb']
    total = 0
    for nb_name in notebooks:
        nb_path = (here / nb_name).resolve()
        if not nb_path.exists():
            print(f'skip: {nb_name} (not found)')
            continue
        written = extract(nb_path, out_dir)
        print(f'{nb_name}: wrote {len(written)} PNG(s)')
        for p in written:
            print(f'  {p.name:<40s} {p.stat().st_size:>8,} bytes')
        total += len(written)

    if total == 0:
        sys.exit(2)
