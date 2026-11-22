"""Plot static site/grid metadata only; no weather-field processing or model IO."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from temporal_folds import project_path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--centers', type=project_path,
                        default=ROOT / 'data/CERRA/spatial_2024_full/centers.csv')
    parser.add_argument('--save_dir', type=project_path, default=ROOT / 'result/paper_review')
    args = parser.parse_args()
    centers = pd.read_csv(args.centers)
    unique = centers.drop_duplicates(['center_y', 'center_x'])
    args.save_dir.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    left.scatter(unique.grid_lon, unique.grid_lat, s=15, color='0.3', alpha=.75,
                 label=f'{len(unique)} native center cells')
    left.add_patch(Rectangle((8.5, 49.5), 3, 2, fill=False, linestyle='--', edgecolor='0.4'))
    for point, offset in [(0, (8, -14)), (45, (8, 6)), (99, (8, 6))]:
        row = centers[centers.point_id == point].iloc[0]
        left.scatter(row.grid_lon, row.grid_lat, s=36, color='tab:red', zorder=4)
        left.annotate(f'ID {point}', (row.grid_lon, row.grid_lat), xytext=offset,
                      textcoords='offset points', fontsize=9)
    left.set(xlabel='Longitude (degrees E)', ylabel='Latitude (degrees N)',
             title='Fixed study sites', xlim=(8.35, 11.65), ylim=(49.4, 51.6))
    left.set_aspect(1 / np.cos(np.deg2rad(50.5)))
    left.grid(alpha=.2)
    left.legend(loc='upper right', fontsize=8)
    local = np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)])
    far = np.array([(dy, dx) for dy in (-10, 0, 10) for dx in (-10, 0, 10) if (dy, dx) != (0, 0)])
    right.scatter(local[:, 1], local[:, 0], marker='s', s=30, color='0.65', label='S02: 5 x 5')
    right.add_patch(Rectangle((-2.5, -2.5), 5, 5, fill=False, edgecolor='0.5'))
    right.add_patch(Rectangle((-1.5, -1.5), 3, 3, fill=False, edgecolor='tab:blue',
                              linewidth=2, label='S01: 3 x 3'))
    right.scatter(far[:, 1], far[:, 0], marker='^', s=60, color='tab:orange',
                  label='A01: eight distant cells')
    right.scatter([0], [0], s=60, color='tab:red', zorder=5, label='Central target (all groups)')
    right.set(xlabel='Native-array dx (grid steps)', ylabel='Native-array dy (grid steps)',
              title='Per-forecast spatial context', xlim=(-12, 12), ylim=(-12, 12),
              xticks=[-10, -2, 0, 2, 10], yticks=[-10, -2, 0, 2, 10])
    right.set_aspect('equal')
    right.grid(alpha=.2)
    right.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, fontsize=8)
    fig.suptitle('Study sites and spatial input configurations', fontsize=13)
    for suffix in ('png', 'svg'):
        fig.savefig(args.save_dir / f'spatial_sampling_setup.{suffix}', dpi=250)
    plt.close(fig)
    print(f'Saved spatial setup diagrams for {len(centers)} IDs / {len(unique)} grids to {args.save_dir}')


if __name__ == '__main__':
    main()
