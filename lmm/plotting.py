def manhattan_mod_final(
        dset='imagenet',
        pcs=range(10),
        pval='P_BOLT_LMM_INF',
        verbose=True,
        save_str=None,
        plot=True,
        lower_lim=None,
        upper_ylm=None,
        imputed=True,
        main_marker='.',
        y_stepsize=5,
        ext='pdf',
        ):
    dfs = []
    for pc in pcs:
        if verbose:
            print(pc)
        if imputed:
            s = f'/mnt/home/Matthias.Kirchler/transferGWAS/lmm/results/{dset}_adjusted/PC_{pc}.imp.txt'
        else:
            s = f'/mnt/home/Matthias.Kirchler/transferGWAS/lmm/results/{dset}_adjusted/PC_{pc}.txt'
        df = pd.read_csv(s, sep='\t')
        df[pval] = df[pval].astype(np.float64)
        dfs.append(df)
    pvs = np.array([df[pval].values for df in dfs])
 
    df = dfs[0]
    df['bonferroni_min'] = bonferroni_min(pvs.T)
    df['which_pc'] = pvs.T.argmin(1)
    
    if plot:
        markers = None
        if lower_lim is None:
            lower_lim = 1e-320
        if not upper_ylm is None:
            if upper_ylm < 1:
                ylm = [0, -np.log10(upper_ylm)+1, y_stepsize]
            else:
                ylm = [0, upper_ylm+1, y_stepsize]
        else:
            ylm = [0, -np.log10(pvs[pvs>lower_lim].min())+1, y_stepsize]

        visuz.marker.mhat(df=df.copy(), chr='CHR', pv='bonferroni_min', bp='BP',
                show=True,
                dim=(0.95*25, 0.95*16),
                gwas_sign_line=True,
                color=("#d7d1c9", "#696464"),
                plot_extra='which_pc',
                ylm=ylm,
                markernames=markers,
                markeridcol='SNP',
                save_str=save_str,
                figtype=ext,
                lower_lim=lower_lim,
                main_marker=main_marker,
                ar=0,
                gfont=12,
                grot=30,
                axlabelfontsize=20,
                axtickfontsize=16,
                )
    return df

def bonferroni_min(p_values):
    return (p_values.shape[1] * p_values.min(1)).clip(max=1.)