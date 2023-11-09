#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np
from copy import copy
from tqdm import tqdm
from ace_pipe.qualitystatistics import QualityStatistics
from ace_pipe.msdata import MSData
from argparse import ArgumentParser
import astropy.stats as astrostats
import matplotlib.pyplot as plt

parser = ArgumentParser(description="Script to flag bad timestamps of MS(es) based on QualityStatistics thresholding.")

parser.add_argument('--ms', help='Input MSs name', type=str, nargs='+')
parser.add_argument('--nsigma_rfi', type=float, required=False, default=5,
                    help="Sigma threshold for RFI percentage. Skip thresholding if set to zero.")
parser.add_argument('--nsigma_mean', type=float, required=False, default=5,
                    help="Sigma threshold for visibility mean. Skip thresholding if set to zero.")
parser.add_argument('--nsigma_var', type=float, required=False, default=5,
                    help="Sigma threshold for visibility variance. Skip thresholding if set to zero.")
parser.add_argument('--nsigma_Dmean', type=float, required=False, default=None,
                    help="Sigma threshold for differential visibility mean. Skip thresholding if set to zero.")
parser.add_argument('--nsigma_Dvar', type=float, required=False, default=None,
                    help="Sigma threshold for differential visibility variance. Skip thresholding if set to zero.")
parser.add_argument('--save_flags_to_h5', type=str, required=False, default=None,
                    help="Save pre-thresholding flags to h5. A filename must be provided when using this parameter.")
parser.add_argument('--dry_run', help='Only report the results from thresholding and save figure(s). Do not update '
                                      'the flags in the MS (Default: False)', action='store_true')


def plot_statistics(input_stats, flagged_stats, times, label, figure_name, uselogy=False):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    axs.plot(times, input_stats, c='b', lw=2, label="Before flagging")
    axs.plot(times, flagged_stats, c='r', lw=2, label="After flagging")

    if uselogy:
        axs.set_yscale('log')

    axs.set_ylim(0.5 * np.amin(input_stats), 1.5 * np.amax(input_stats))
    axs.set_ylabel(label, fontsize=16)
    axs.set_xlabel(r'Time [min]', fontsize=16)
    axs.tick_params(axis='both', direction='in', which='both', labelsize=16)
    axs.set_title("Flagged times: {}".format(abs(np.sum(flagged_stats.mask))), fontsize=18)

    fig.savefig(figure_name, bbox_inches='tight', dpi=150, transparent=False, facecolor='white')


def calculate_statistics(qstats, nsigma_rfi, nsigma_mean, nsigma_var, nsigma_Dmean, nsigma_Dvar):
    times = (qstats.times - qstats.times[0]) / 60.

    usamples = np.ma.array(abs(getattr(qstats, 'time_stats')['Count']).mean(axis=1))
    usamples[usamples == 0] = np.ma.masked
    usamples[np.isnan(usamples)] = np.ma.masked

    if nsigma_Dmean is not None or nsigma_Dvar is not None:
        Dusamples = np.ma.array(abs(getattr(qstats, 'time_stats')['DCount']).mean(axis=1))
        Dusamples[Dusamples == 0] = np.ma.masked
        Dusamples[np.isnan(Dusamples)] = np.ma.masked

    if nsigma_rfi is not None:
        rfi_perc = np.ma.array(qstats.get_rfi_percentage()['time_stats'].mean(axis=1))
        rfi_perc.mask = copy(usamples.mask)
        clipped_rfi_perc = astrostats.sigma_clip(rfi_perc, sigma=nsigma_rfi, cenfunc='median', stdfunc='mad_std')
        plot_statistics(rfi_perc, clipped_rfi_perc, times, r'RFI percentage', 'rfi_percentage.png')

    if nsigma_mean is not None:
        mean = np.ma.array(qstats.get_mean()['time_stats'].mean(axis=1))
        mean.mask = copy(usamples.mask)
        clipped_mean = astrostats.sigma_clip(mean, sigma=nsigma_mean, cenfunc='median', stdfunc='mad_std')
        plot_statistics(mean, clipped_mean, times, r'Visibility mean', 'mean.png', uselogy=True)

    if nsigma_var is not None:
        variance = np.ma.array(qstats.get_variance()['time_stats'].mean(axis=1))
        variance.mask = copy(usamples.mask)
        clipped_var = astrostats.sigma_clip(variance, sigma=nsigma_var, cenfunc='median', stdfunc='mad_std')
        plot_statistics(variance, clipped_var, times, r'Visibility Variance', 'variance.png', uselogy=True)

    if nsigma_Dmean is not None:
        Dmean = np.ma.array(qstats.get_differential_mean()['time_stats'].mean(axis=1))
        Dmean.mask = copy(Dusamples.mask)
        clipped_Dmean = astrostats.sigma_clip(Dmean, sigma=nsigma_Dmean, cenfunc='median', stdfunc='mad_std')
        plot_statistics(Dmean, clipped_Dmean, times, r'Visibility Variance', 'differential_mean.png', uselogy=True)

    if nsigma_Dvar is not None:
        Dvar = np.ma.array(qstats.get_differential_variance()['time_stats'].mean(axis=1))
        Dvar.mask = copy(Dusamples.mask)
        clipped_Dvar = astrostats.sigma_clip(Dvar, sigma=nsigma_Dvar, cenfunc='median', stdfunc='mad_std')
        plot_statistics(Dvar, clipped_Dvar, times, r'Visibility Variance', 'differential_variance.png', uselogy=True)

    print("Figures saved.")


def write_flags(msname, bad_timestamps, save_flags_to_h5):
    msdata = MSData.load(msname, read_flags=True)

    for i, t_idx in enumerate(tqdm(bad_timestamps)):
        msdata.flags[t_idx, :, :, :] = 1


def main():
    args = parser.parse_args(sys.argv[1:])
    mslist = args.ms

    nsigma_rfi = args.nsigma_rfi
    nsigma_mean = args.nsigma_mean
    nsigma_var = args.nsigma_var
    nsigma_Dmean = args.sigma_Dmean
    nsigma_Dvar = args.nsigma_Dvar
    dry_run = args.dry_run

    for msname in mslist:
        qstats = QualityStatistics.load_from_ms(msname)

        if dry_run:
            calculate_statistics(qstats, nsigma_rfi, nsigma_mean, nsigma_var, nsigma_Dmean, nsigma_Dvar)
        else:
            calculate_statistics(qstats, nsigma_rfi, nsigma_mean, nsigma_var, nsigma_Dmean, nsigma_Dvar)
            bad_timestamps = qstats.get_bad_indices('time_stats', nsigma_rfi_thresh=nsigma_rfi,
                                                    nsigma_mean=nsigma_mean, nsigma_var=nsigma_var,
                                                    nsigma_Dmean=nsigma_Dmean, nsigma_Dvar=nsigma_Dvar)


if __name__ == '__main__':
    main()
