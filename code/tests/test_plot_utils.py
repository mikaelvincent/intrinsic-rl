from __future__ import annotations

from pathlib import Path


def test_plot_utils_apply_rcparams_and_save(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic

    rc_before = dict(plt.rcParams)
    try:
        plt2 = apply_rcparams_paper()

        assert float(plt2.rcParams["figure.dpi"]) == 150.0
        assert float(plt2.rcParams["savefig.dpi"]) == 300.0
        assert float(plt2.rcParams["font.size"]) == 9.0

        fig, ax = plt2.subplots(figsize=(2, 2))
        ax.plot([0.0, 1.0], [0.0, 1.0])

        out = tmp_path / "plot.png"
        tmp_out = out.with_suffix(out.suffix + ".tmp")

        save_fig_atomic(fig, out)

        assert out.exists()
        assert out.stat().st_size > 0
        assert not tmp_out.exists()

        plt2.close(fig)
    finally:
        plt.rcParams.update(rc_before)
