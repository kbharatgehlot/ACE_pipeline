import numpy as np
from tqdm import tqdm
import utils
import astropy.stats as stats


def get_bad_baselines(quality_stats, nsigma_rfi_thresh=None, nsigma_mean=None, nsigma_var=None, nsigma_Dmean=None,
                      nsigma_Dvar=None):
    """ Get a list of bad baselines based on sigma thresholding of RFI percentage, Mean, Variance, differential mean,
        and differential variance. Thresholding is performed only for the statistics whose nsigma is provided.

        parameters
        ----------
        quality_stats : QualityStatistics object
        nsigma_rfi_thresh : int or float
            sigma threshold (Default: None) for rfi percentage.
            median - nsigma*mad_std < samples > median + nsigma*mad_std are flagged as bad
        nsigma_mean : int or float
            sigma threshold  (Default: None) for mean.
            median - nsigma*std < samples > median + nsigma*std are flagged as bad
        nsigma_var : int or float
            sigma threshold  (Default: None) for variance.
            median - nsigma*std < samples > median + nsigma*std are flagged as bad
        nsigma_Dmean : int or float
            sigma threshold  (Default: None) for differential mean.
            median - nsigma*std < samples > median + nsigma*std are flagged as bad
        nsigma_Dvar : int or float
            sigma threshold  (Default: None) for differential variance.
            median - nsigma*std < samples > median + nsigma*std are flagged as bad

        returns
        ----------
        bad_baselines: list of tuples (int,int)
        List of bad antenna pairs [(ant1,ant2),...]"""

    ant1 = quality_stats.ant1
    ant2 = quality_stats.ant2

    # I average over all correlation pairs (axis=1).

    usamples = np.ma.array(abs(quality_stats.bl_stats['Count']).mean(axis=1))
    usamples[ant1 == ant2] = np.ma.masked
    usamples[usamples == 0] = np.ma.masked
    usamples[np.isnan(usamples)] = np.ma.masked
    init_mask = usamples.mask

    Dusamples = np.ma.array(abs(quality_stats.bl_stats['Count']).mean(axis=1))
    Dusamples[ant1 == ant2] = np.ma.masked
    Dusamples[Dusamples == 0] = np.ma.masked
    Dusamples[np.isnan(Dusamples)] = np.ma.masked
    init_Dmask = Dusamples.mask

    flag_mask = usamples.mask

    if nsigma_rfi_thresh is not None:
        rfi_perc = np.ma.array(abs(utils.get_rfi_percentage(quality_stats)['bl_stats']).mean(axis=1))
        rfi_perc.mask = init_mask
        clipped_rfi_perc = stats.sigma_clip(rfi_perc, sigma=nsigma_rfi_thresh, cenfunc='median', stdfunc='mad_std')
        flag_mask = np.logical_xor(clipped_rfi_perc.mask, flag_mask)

    if nsigma_mean is not None:
        mean = np.ma.array(abs(utils.get_mean(quality_stats)['bl_stats']).mean(axis=1))
        mean.mask = init_mask
        clipped_mean = stats.sigma_clip(mean, sigma=nsigma_mean, cenfunc='median', stdfunc='std')
        flag_mask = np.logical_xor(clipped_mean.mask, flag_mask)

    if nsigma_var is not None:
        variance = np.ma.array(abs(utils.get_variance(quality_stats)['bl_stats']).mean(axis=1))
        variance.mask = init_mask
        clipped_var = stats.sigma_clip(variance, sigma=nsigma_var, cenfunc='median', stdfunc='std')
        flag_mask = np.logical_xor(clipped_var.mask, flag_mask)

    if nsigma_Dmean is not None:
        Dmean = np.ma.array(abs(utils.get_differential_mean(quality_stats)['bl_stats']).mean(axis=1))
        Dmean.mask = init_Dmask
        clipped_Dmean = stats.sigma_clip(Dmean, sigma=nsigma_Dmean, cenfunc='median', stdfunc='std')
        flag_mask = np.logical_xor(clipped_Dmean.mask, flag_mask)

    if nsigma_Dvar is not None:
        Dvar = np.ma.array(abs(utils.get_differential_variance(quality_stats)['bl_stats']).mean(axis=1))
        Dvar.mask = init_Dmask
        clipped_Dvar = stats.sigma_clip(Dvar, sigma=nsigma_Dvar, cenfunc='median', stdfunc='std')
        flag_mask = np.logical_xor(clipped_Dvar.mask, flag_mask)

    bad_baselines = []
    for k, v in tqdm(zip(ant1[flag_mask == True], ant2[flag_mask == True])):
        bad_baselines.append((k, v))

    return bad_baselines


def get_bad_times(quality_stats, nsigma_rfi_thresh=None, nsigma_mean=None, nsigma_var=None, nsigma_Dmean=None,
                  nsigma_Dvar=None):
    """ Get a list of bad time indices based on sigma thresholding of RFI percentage, Mean, Variance, differential mean,
          and differential variance. Thresholding is performed only for the statistics whose nsigma is provided.

          parameters
          ----------
          quality_stats : QualityStatistics object
          nsigma_rfi_thresh : int or float
              sigma threshold (Default: None) for rfi percentage.
              median - nsigma*mad_std < samples > median + nsigma*mad_std are flagged as bad
          nsigma_mean : int or float
              sigma threshold  (Default: None) for mean.
              median - nsigma*std < samples > median + nsigma*std are flagged as bad
          nsigma_var : int or float
              sigma threshold  (Default: None) for variance.
              median - nsigma*std < samples > median + nsigma*std are flagged as bad
          nsigma_Dmean : int or float
              sigma threshold  (Default: None) for differential mean.
              median - nsigma*std < samples > median + nsigma*std are flagged as bad
          nsigma_Dvar : int or float
              sigma threshold  (Default: None) for differential variance.
              median - nsigma*std < samples > median + nsigma*std are flagged as bad

          returns
          ----------
          bad_times_idx: list of int
          List of bad times indices [idx1,idx2,...]"""

    # I average over all correlation pairs (axis=1).
    usamples = np.ma.array(abs(quality_stats.time_stats['Count']).mean(axis=1))
    usamples[usamples == 0] = np.ma.masked
    usamples[np.isnan(usamples)] = np.ma.masked
    init_mask = usamples.mask

    Dusamples = np.ma.array(abs(quality_stats.time_stats['Count']).mean(axis=1))
    Dusamples[Dusamples == 0] = np.ma.masked
    Dusamples[np.isnan(Dusamples)] = np.ma.masked
    init_Dmask = Dusamples.mask

    flag_mask = usamples.mask

    if nsigma_rfi_thresh is not None:
        rfi_perc = np.ma.array(abs(utils.get_rfi_percentage(quality_stats)['time_stats']).mean(axis=1))
        rfi_perc.mask = init_mask
        clipped_rfi_perc = stats.sigma_clip(rfi_perc, sigma=nsigma_rfi_thresh, cenfunc='median', stdfunc='mad_std')
        flag_mask = np.logical_xor(clipped_rfi_perc.mask, flag_mask)

    if nsigma_mean is not None:
        mean = np.ma.array(abs(utils.get_mean(quality_stats)['time_stats']).mean(axis=1))
        mean.mask = init_mask
        clipped_mean = stats.sigma_clip(mean, sigma=nsigma_mean, cenfunc='median', stdfunc='std')
        flag_mask = np.logical_xor(clipped_mean.mask, flag_mask)

    if nsigma_var is not None:
        variance = np.ma.array(abs(utils.get_variance(quality_stats)['time_stats']).mean(axis=1))
        variance.mask = init_mask
        clipped_var = stats.sigma_clip(variance, sigma=nsigma_var, cenfunc='median', stdfunc='std')
        flag_mask = np.logical_xor(clipped_var.mask, flag_mask)

    if nsigma_Dmean is not None:
        Dmean = np.ma.array(abs(utils.get_differential_mean(quality_stats)['time_stats']).mean(axis=1))
        Dmean.mask = init_Dmask
        clipped_Dmean = stats.sigma_clip(Dmean, sigma=nsigma_Dmean, cenfunc='median', stdfunc='std')
        flag_mask = np.logical_xor(clipped_Dmean.mask, flag_mask)

    if nsigma_Dvar is not None:
        Dvar = np.ma.array(abs(utils.get_differential_variance(quality_stats)['time_stats']).mean(axis=1))
        Dvar.mask = init_Dmask
        clipped_Dvar = stats.sigma_clip(Dvar, sigma=nsigma_Dvar, cenfunc='median', stdfunc='std')
        flag_mask = np.logical_xor(clipped_Dvar.mask, flag_mask)

    bad_times_idx = np.where(flag_mask == True)[0]

    return bad_times_idx
