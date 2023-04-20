import h5py
from tqdm import tqdm


def qstat_rfi_percentage(RFICount, Count):
    """Calculate RFI percentage:
        Parameters
        ----------
            RFICount: ndarray
                RFI flag count for a given statistic
            Count: ndarray
        :return:
            rfi_percentage: rfi_percentage for a given statistic
        """
    return 100. * RFICount.real / (RFICount.real + Count.real)


def qstat_mean(Sum, Count):
    """Calculate Mean from a quality statistic:
        Parameters
        ----------
            Sum: ndarray
            Count: ndarray
        :return:
            mean: mean for a given statistic """
    return abs(Sum / Count.real)


def qstat_variance(Sum, SumP2, Count):
    """Calculate variance from a quality statistic:
            Parameters
            ----------
                Sum: ndarray
                SumP2: ndarray
                Count: ndarray
            :return:
                Variance: variance for a given statistic """
    return abs((SumP2 - (Sum * Sum / Count.real)) / (Count.real - 1))


def qstat_differential_mean(DSum, DCount):
    """Calculate differential mean from a quality statistic:
            Parameters
            ----------
                DSum: ndarray
                DCount: ndarray
            :return:
                differential_mean: differential mean for a given statistic """
    return abs(DSum / DCount.real)


def qstat_differential_variance(DSum, DSumP2, DCount):
    """Calculate differential variance from a quality statistic:
            Parameters
            ----------
                Sum: ndarray
                Count: ndarray
            :return:
                mean: mean for a given statistic """
    return abs((DSumP2 - (DSum * DSum / DCount.real)) / (DCount.real - 1))


def save_moments_to_h5(quality_stats, filename, overwrite=False):
    """Save the moments {rfi_percentage, mean, variance, differential mean, differential variance} to a h5 file."""

    try:
        assert isinstance(filename, str)
        if overwrite:
            h5_file = h5py.File(filename, 'w')
        else:
            h5_file = h5py.File(filename, 'w-')

        meta_group = h5_file.create_group('metadata')
        meta_group.create_dataset('stats_type', data=quality_stats.stats_type)
        meta_group.create_dataset('ant1', data=quality_stats.ant1)
        meta_group.create_dataset('ant2', data=quality_stats.ant2)
        meta_group.create_dataset('times', data=quality_stats.times)
        meta_group.create_dataset('freqs', data=quality_stats.freqs)

        moments_name = ['rfi_percentage', 'mean', 'variance', 'Dmean', 'Dvariance']
        meta_group.create_dataset('moments_name', data=moments_name)

        bl_stats_group = h5_file.create_group(quality_stats.stats_type[0])  # stats_type[0]='bl_stats'
        time_stats_group = h5_file.create_group(quality_stats.stats_type[1])  # stats_type[1]='time_stats'
        freq_stats_group = h5_file.create_group(quality_stats.stats_type[2])  # stats_type[2]='freq_stats'

        rfi_percent, mean, variance, Dmean, Dvariance = quality_stats.convert_to_moments()
        moments_list = [rfi_percent, mean, variance, Dmean, Dvariance]

        for i, moment in enumerate(tqdm(moments_list, desc='Writing moments to file: {}'.format(filename))):
            bl_stats_group.create_dataset(moments_name[i], data=moment[quality_stats.stats_type[0]])
            time_stats_group.create_dataset(moments_name[i], data=moment[quality_stats.stats_type[1]])
            freq_stats_group.create_dataset(moments_name[i], data=moment[quality_stats.stats_type[2]])

        print("Moments written to: {}".format(filename))

    except AssertionError:
        print("filename must be a string.")
        raise


def load_moments_from_h5(filename):
    try:
        with h5py.File(filename, 'r') as h5_file:
            metadata = h5_file.get('metadata')

            stats_type = [i.decode() for i in metadata.get('stats_type')]
            moments_name = [i.decode() for i in metadata.get('moments_name')]

            ant1 = metadata.get('ant1')[:]
            ant2 = metadata.get('ant2')[:]
            times = metadata.get('times')[:]
            freqs = metadata.get('freqs')[:]

            bl_moments = {}
            time_moments = {}
            freq_moments = {}

            bl_stat_table = h5_file.get(stats_type[0])
            time_stat_table = h5_file.get(stats_type[1])
            freq_stat_table = h5_file.get(stats_type[2])

            for i, name in enumerate(tqdm(moments_name, desc='Reading moments from file:{}'.format(filename))):
                bl_moments[name[i]] = bl_stat_table[name[i]][:]
                time_moments[name[i]] = time_stat_table[name[i]][:]
                freq_moments[name[i]] = freq_stat_table[name[i]][:]

        return moments_name, bl_moments, time_moments, freq_moments, ant1, ant2, times, freqs

    except FileNotFoundError:
        print('{} file does not exist'.format(filename))
        raise
