import click
import numpy as np
import pandas as pd
import seaborn as sns
from bioinfokit import visuz
from matplotlib import pyplot as plt


@click.command()
@click.option("--infile", type=str)
@click.option("--outfile_template", type=str)
@click.option("--pval_name", type=str)
def main(infile, pval_name, outfile_template):
    df = pd.read_csv(infile, sep="\t")
    df[pval_name] = df[pval_name].astype(np.float64)
    qq_plot(df[pval_name], fn=outfile_template + "_qq.png")
    mhat_plot(df, pval_name, fn=outfile_template + "_mhat")


def mhat_plot(
    df,
    pval,
    fn,
    ext="png",
    lower_lim=None,
    upper_ylm=None,
    y_stepsize=5,
):
    """
    TODO: better styling of mhat
    """
    markers = None
    if lower_lim is None:
        lower_lim = 1e-320
    if not upper_ylm is None:
        if upper_ylm < 1:
            ylm = [0, -np.log10(upper_ylm) + 1, y_stepsize]
        else:
            ylm = [0, upper_ylm + 1, y_stepsize]
    else:
        mn = df[df[pval] > lower_lim][pval].min()
        ylm = [0, -np.log10(mn) + 1, y_stepsize]
    visuz.marker.mhat(
        df=df.copy(),
        chr="CHR",
        pv=pval,
        # bp="BP",
        # show=True,
        dim=(0.95 * 25, 0.95 * 16),
        gwas_sign_line=True,
        gwasp=5E-09,
        color=("#d7d1c9", "#696464"),
        # plot_extra='which_pc',
        ylm=ylm,
        markernames=markers,
        markeridcol="SNP",
        figname=fn,
        figtype=ext,
        # lower_lim=lower_lim,
        # main_marker=".",
        ar=0,
        gfont=12,
        # grot=30,
        axlabelfontsize=20,
        axtickfontsize=16,
    )
    plt.close()


def qq_plot(pvals, fn, lam=None, size=(10, 10), cutoff=1e-3, dpi=400):
    sns.set_style("whitegrid")
    font_small = 16
    font_large = 20
    p = np.array(sorted(pvals))
    p[p == 0] = 5e-324
    n = len(p)
    x = np.arange(1, n + 1) / n
    upto = np.where(x > cutoff)[0].min()
    p = -np.log10(p)
    x = -np.log10(x)
    fig = plt.figure(figsize=size)
    plt.plot(x[upto:], p[upto:], linewidth=3, linestyle="-", rasterized=True)
    plt.scatter(x[:upto], p[:upto], marker=".", rasterized=True)
    plt.plot(x, x, c="r", linewidth=2, linestyle="--", rasterized=True)
    plt.xlabel("Expected $-\log_{10}(p)$", fontsize=font_large)
    plt.ylabel("Observed $-\log_{10}(p)$", fontsize=font_large)
    plt.xticks(size=font_small)
    plt.yticks(size=font_small)
    if lam:
        yloc = 0.95 * p.max()
        plt.text(
            0.1,
            yloc,
            f"$\lambda = {lam:.3f}$",
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=font_large,
        )
    if fn:
        plt.savefig(fn, dpi=dpi)
    plt.show()
    plt.close()
    sns.reset_orig()


if __name__ == "__main__":
    main()
