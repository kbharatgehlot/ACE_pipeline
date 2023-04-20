# noinspection PyPackageRequirements
import os
from typing import List, Tuple, Any

import numpy as np
import casacore.tables as tbl
from tqdm import tqdm, trange
import h5py
import warnings
from copy import copy, deepcopy
from ACE_pipe import utils
import astropy.stats as astrostats


class QualityStatistics(object):
    """
    Object/class to read/store/manipulate contents of quality statistics from aoflagger/aoquality.
    Quality statistics include: ['RFICount', 'Count', 'Sum', 'SumP2', 'DCount', 'DSum', 'DSumP2']
    RFICount: flagged samples,
    Count: unflagged samples,
    Sum: sum of unflagged samples,
    SumP2: sum of square of unflagged samples,
    Dcount: differential count,
    DSum: differential sum,
    DSumP2: differential square sum
    "D" in front is "differential", i.e.it is the channel values minus the next channel values.

    These quality statistics can be used to calculate more useful statistics, e.g.:
    Mean = Sum/Count
    Variance = (sumP2 - sum * sum /count) / (count - 1.0)
    """

    def __init__(self, stats_name, stats_kind, stats_type, bl_stats, time_stats, freq_stats, ant1, ant2, times, freqs):
        """Create a new QualityStatistics object.
        Attributes
        ----------
        stats_name : dict (str)
            dictionary with the names of statistics
        stats_kind : array (int)
            index associated with the statistics
        stats_type : list (str)
            type of statistic (baseline, time, frequency)
        bl_stats : dict
            dictionary with baseline statistics
        time_stats : dict
            dictionary with time statistics
        freq_stats : dict
            dictionary with freq statistics
        ant1 : array (int)
            antenna1 index
        ant2 : array (int)
            antenna2 index
        times : array (float)
            array of times, modified julian date in units of seconds
        freqs : array (float)
            array of channel frequencies, in units of Hz
        """

        self.stats_name = stats_name
        self.stats_kind = stats_kind
        self.stats_type = stats_type
        self.bl_stats = bl_stats
        self.time_stats = time_stats
        self.freq_stats = freq_stats
        self.ant1 = ant1
        self.ant2 = ant2
        self.times = times
        self.freqs = freqs

    @staticmethod
    def load_from_ms(msname):
        """Function to load the QualityStatistics from the measurement set. The default statistics in an MS are
        produced either by aoflagger or aoquality tools.
        Parameters
        ----------
            msname : str Name of the MS.
        :return:
            QualityStatistics(stats_name, stats_kind, stats_type, bl_stats, time_stats, freq_stats, ant1, ant2, times,
            freqs)
        """

        if not os.path.exists(msname):
            raise FileNotFoundError('MS not found in the provided path.')
        elif not os.path.exists(msname + "/QUALITY_KIND_NAME"):
            raise FileNotFoundError('Quality statistics not found in the MS.')

        # Dict object for name of the statistic and corresponding kind
        stats_name = {}
        with tbl.table(msname + "/QUALITY_KIND_NAME", readonly=True) as statobj:
            stats_kind = statobj.getcol('KIND')
            for i, kind in enumerate(stats_kind):
                stats_name[kind] = statobj.getcol('NAME', startrow=i, nrow=1)[0]

        # Dict object for baseline statistics
        print("Reading baseline quality statistics...")
        bl_stats = {}
        with tbl.table(msname + "/QUALITY_BASELINE_STATISTIC", readonly=True) as statobj:
            ant1 = statobj.getcol('ANTENNA1').reshape(-1, len(stats_kind))[:, 0]
            ant2 = statobj.getcol('ANTENNA2').reshape(-1, len(stats_kind))[:, 0]
            values = statobj.getcol('VALUE').reshape(-1, len(stats_kind), 4)
            for i, kind in enumerate(stats_kind):
                bl_stats[stats_name[kind]] = values[:, i, :]

        # Dict object for frequency statistics
        print("Reading frequency quality statistics...")
        freq_stats = {}
        with tbl.table(msname + "/QUALITY_FREQUENCY_STATISTIC", readonly=True) as statobj:
            freqs = np.unique(statobj.getcol('FREQUENCY'))
            values = statobj.getcol('VALUE').reshape(-1, len(stats_kind), 4)
            for i, kind in enumerate(stats_kind):
                freq_stats[stats_name[kind]] = values[:, i, :]

        # Dict object for time statistics
        print("Reading time quality statistics...")
        time_stats = {}
        with tbl.table(msname + "/QUALITY_TIME_STATISTIC", readonly=True) as statobj:
            times = np.unique(statobj.getcol('TIME'))
            values = statobj.getcol('VALUE').reshape(-1, len(stats_kind), 4)
            for i, kind in enumerate(stats_kind):
                time_stats[stats_name[kind]] = values[:, i, :]

        stats_type = ['bl_stats', 'time_stats', 'freq_stats']
        return QualityStatistics(stats_name, stats_kind, stats_type, bl_stats, time_stats, freq_stats, ant1, ant2,
                                 times, freqs)

    @staticmethod
    def load_from_h5(filename):
        """Function to load the QualityStatistics from a h5 file that was written using the save_statistics_to_h5()
        method of the QualityStatistics.
        Parameters
        ----------
        filename : str
            Name of the h5 file.
        :return:
            QualityStatistics(stats_name, stats_kind, stats_type, bl_stats, time_stats, freq_stats, ant1, ant2, times,
            freqs)

        """
        assert isinstance(filename, str), "filename must be a string."

        try:
            with h5py.File(filename, 'r') as h5_file:
                metadata = h5_file.get('metadata')

                stats_kind = metadata.get('stats_kind')[:]
                ant1 = metadata.get('ant1')[:]
                ant2 = metadata.get('ant2')[:]
                times = metadata.get('times')[:]
                freqs = metadata.get('freqs')[:]
                stats_type = [i.decode() for i in metadata.get('stats_type')]
                name = [i.decode() for i in metadata.get('stats_name')]

                stats_name = {}
                bl_stats = {}
                freq_stats = {}
                time_stats = {}

                bl_stat_table = h5_file.get('bl_stats')
                time_stat_table = h5_file.get('time_stats')
                freq_stat_table = h5_file.get('freq_stats')

                for i, kind in enumerate(
                        tqdm(stats_kind, desc='Reading quality statistics from file:{}'.format(filename))):
                    stats_name[kind] = name[i]
                    bl_stats[name[i]] = bl_stat_table[name[i]][:]
                    time_stats[name[i]] = time_stat_table[name[i]][:]
                    freq_stats[name[i]] = freq_stat_table[name[i]][:]

        except FileNotFoundError:
            print('h5 file not found in the provided path.')

        return QualityStatistics(stats_name, stats_kind, stats_type, bl_stats, time_stats, freq_stats, ant1, ant2,
                                 times, freqs)

    def save_to_h5(self, filename, overwrite=False):
        """Save the quality statistics to a h5 file.
        Parameters
        ----------
        filename : str
            Name of the h5 file.
        overwrite : bool (default: False)
            overwrite the existing file, if True.
        """
        assert isinstance(filename, str), "filename must be a string."
        if overwrite:
            warnings.warn("overwrite=True, {} will be overwritten".format(filename), UserWarning)
            h5_file = h5py.File(filename, 'w')
        else:
            h5_file = h5py.File(filename, 'w-')

        meta_group = h5_file.create_group('metadata')
        meta_group.create_dataset('stats_kind', data=self.stats_kind)

        meta_group.create_dataset('ant1', data=self.ant1)
        meta_group.create_dataset('ant2', data=self.ant2)
        meta_group.create_dataset('times', data=self.times)
        meta_group.create_dataset('freqs', data=self.freqs)
        meta_group.create_dataset('stats_type', data=self.stats_type)

        bl_stats_group = h5_file.create_group('bl_stats')
        freq_stats_group = h5_file.create_group('freq_stats')
        time_stats_group = h5_file.create_group('time_stats')

        stat_name = []
        for i, kind in enumerate(tqdm(self.stats_kind, desc='Writing quality statistics to file: {}'.format(filename))):
            name = self.stats_name[kind]
            stat_name.append(str(name))
            bl_stats_group.create_dataset(name, data=self.bl_stats[name])
            freq_stats_group.create_dataset(name, data=self.freq_stats[name])
            time_stats_group.create_dataset(name, data=self.time_stats[name])

        meta_group.create_dataset('stats_name', data=stat_name)
        print("Quality statistics saved to {}".format(filename))
        return

    def get_rfi_percentage(self):
        """Calculate rfi percentage for baseline, time, and frequency quality statistics.
        :return:
            rfi_percentage: Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)}
        """

        rfi_percentage = {}
        for stats in self.stats_type:
            stat_obj = getattr(self, stats)
            RFICount = stat_obj['RFICount']
            Count = stat_obj['Count']
            rfi_percentage[stats] = utils.qstat_rfi_percentage(RFICount, Count)

        return rfi_percentage

    def get_mean(self):
        """Calculate mean statistic for baseline, time, and frequency quality statistics.
        :return:
            mean: Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)}
         """
        mean = {}
        for stats in self.stats_type:
            stat_obj = getattr(self, stats)
            Sum = stat_obj['Sum']
            Count = stat_obj['Count']
            mean[stats] = utils.qstat_mean(Sum, Count)

        return mean

    def get_variance(self):
        """Calculate variance statistic for baseline, time, and frequency quality statistics.
        :return:
            variance: Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)}
         """

        variance = {}
        for stats in self.stats_type:
            stat_obj = getattr(self, stats)
            SumP2 = stat_obj['SumP2']
            Sum = stat_obj['Sum']
            Count = stat_obj['Count']
            variance[stats] = utils.qstat_variance(Sum, SumP2, Count)

        return variance

    def get_differential_mean(self):
        """Calculate differential mean statistic for baseline, time, and frequency quality statistics.
        :return:
            differential_mean : Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)}
         """

        mean = {}
        for stats in self.stats_type:
            stat_obj = getattr(self, stats)
            DSum = stat_obj['DSum']
            DCount = stat_obj['DCount']
            mean[stats] = utils.qstat_differential_mean(DSum, DCount)

        return mean

    def get_differential_variance(self):
        """ Calculate differential mean statistic for baseline, time, and frequency quality statistics.
        :return:
            differential_variance Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)}
        """

        variance = {}
        for stats in self.stats_type:
            stat_obj = getattr(self, stats)
            DSumP2 = stat_obj['DSumP2']
            DSum = stat_obj['DSum']
            DCount = stat_obj['DCount']
            variance[stats] = utils.qstat_differential_variance(DSum, DSumP2, DCount)

        return variance

    def convert_to_moments(self):
        """ Convert the QualityStatistics to moments: rfi_percentage, mean, variance, Dmean, and Dvariance.
        :return:
            rfi_percentage: Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)},
            mean: Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)},
            variance: Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)},
            differential_mean : Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)},
            differential_variance Dict {'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols),
            'freq_stats': ndarray(nfreqs,npols)}
        """

        rfi_perc = self.get_rfi_percentage()
        mean = self.get_mean()
        var = self.get_variance()
        diff_mean = self.get_differential_mean()
        diff_var = self.get_differential_variance()

        return rfi_perc, mean, var, diff_mean, diff_var

    def get_bad_indices(self, stats_type, nsigma_rfi_thresh=None, nsigma_mean=None, nsigma_var=None, nsigma_Dmean=None,
                        nsigma_Dvar=None):
        """Search for bad datapoints based on sigma thresholding of moments for different statistics. Current
        implementation median and mad_std to search for outliers using sigma_clipping. Thresholding is performed only
        for the statistics whose nsigma is provided. If more than one nsigma is provided then the returned indices are a
        combination of bad indices from various moments.
        Parameters
        ----------
        stats_type : str
            Type of the quality statistic. Acceptable values are: 'bl_stats', 'time_stats', freq_stats
        nsigma_rfi_thresh : int or float (default: None)
            sigma threshold for rfi-percentage.
        nsigma_mean : int or float (default: None)
            sigma threshold for mean statistic.
        nsigma_var : int or float (default: None)
            sigma threshold for variance statistic.
        nsigma_dmean : int or float (default: None)
            sigma threshold for differential mean statistic.
        nsigma_dvar : int or float (default: None)
            sigma threshold for differential variance statistic.

        :return:
        bad_indices : list of tuples if stats_type is 'bl_stats' else an array containing bad time/freq indices"""

        assert isinstance(stats_type, str), "stats_type must be a string."
        assert stats_type in self.stats_type, "Acceptable values are: 'bl_stats', 'time_stats', freq_stats"

        usamples = np.ma.array(abs(getattr(self, stats_type)['Count']).mean(axis=1))
        Dusamples = np.ma.array(abs(getattr(self, stats_type)['DCount']).mean(axis=1))

        if stats_type == 'bl_stats':
            usamples[self.ant1 == self.ant2] = np.ma.masked
            Dusamples[self.ant1 == self.ant2] = np.ma.masked

        usamples[usamples == 0] = np.ma.masked
        usamples[np.isnan(usamples)] = np.ma.masked

        Dusamples[Dusamples == 0] = np.ma.masked
        Dusamples[np.isnan(Dusamples)] = np.ma.masked

        init_mask = usamples.mask
        init_Dmask = Dusamples.mask

        flag_mask = copy(usamples.mask)

        if nsigma_rfi_thresh is not None:
            rfi_perc = np.ma.array(self.get_rfi_percentage()[stats_type].mean(axis=1))
            rfi_perc.mask = copy(init_mask)
            clipped_rfi_perc = astrostats.sigma_clip(rfi_perc, sigma=nsigma_rfi_thresh, cenfunc='median',
                                                     stdfunc='mad_std')
            flag_mask = np.logical_xor(clipped_rfi_perc.mask, flag_mask)

        if nsigma_mean is not None:
            mean = np.ma.array(self.get_mean()[stats_type].mean(axis=1))
            mean.mask = copy(init_mask)
            clipped_mean = astrostats.sigma_clip(mean, sigma=nsigma_mean, cenfunc='median', stdfunc='mad_std')
            flag_mask = np.logical_xor(clipped_mean.mask, flag_mask)

        if nsigma_var is not None:
            variance = np.ma.array(self.get_variance()[stats_type].mean(axis=1))
            variance.mask = copy(init_mask)
            clipped_var = astrostats.sigma_clip(variance, sigma=nsigma_var, cenfunc='median', stdfunc='mad_std')
            flag_mask = np.logical_xor(clipped_var.mask, flag_mask)

        if nsigma_Dmean is not None:
            Dmean = np.ma.array(self.get_differential_mean()[stats_type].mean(axis=1))
            Dmean.mask = init_Dmask
            clipped_Dmean = astrostats.sigma_clip(Dmean, sigma=nsigma_Dmean, cenfunc='median', stdfunc='mad_std')
            flag_mask = np.logical_xor(clipped_Dmean.mask, flag_mask)

        if nsigma_Dvar is not None:
            Dvar = np.ma.array(self.get_differential_variance()[stats_type].mean(axis=1))
            Dvar.mask = copy(init_Dmask)
            clipped_Dvar = astrostats.sigma_clip(Dvar, sigma=nsigma_Dvar, cenfunc='median', stdfunc='mad_std')
            flag_mask = np.logical_xor(clipped_Dvar.mask, flag_mask)

        if stats_type == 'bl_stats':
            bad_indices = []
            for k, v in tqdm(zip(self.ant1[flag_mask == True], self.ant2[flag_mask == True])):
                bad_indices.append((k, v))
            return bad_indices
        elif stats_type == 'time_stats':
            bad_times_idx = np.where(flag_mask == True)[0]
            return bad_times_idx
        elif stats_type == 'freq_stats':
            bad_freqs_idx = np.where(flag_mask == True)[0]
            return bad_freqs_idx
        else:
            pass
