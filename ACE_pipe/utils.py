import h5py
from tqdm import tqdm


def get_rfi_percentage(quality_stats):
    """ This function calculates rfi percentage from a  specific type of quality statistic. Currently only one
    statistic is allowed at a time. The method can be invoked multiple times to calculate the quantity for
    multiple statistics. Returns a real valued array """

    rfi_percentage = {}
    for stats in quality_stats.stats_type:
        stat_obj = getattr(quality_stats, stats)
        RFICount = stat_obj['RFICount']
        Count = stat_obj['Count']
        rfi_percentage[stats] = 100. * RFICount.real / (RFICount.real + Count.real)

    return rfi_percentage


def get_mean(quality_stats):
    """ This function calculates mean of a specific type of quality statistic. Currently only one
    statistic is allowed at a time. The method can be invoked multiple times to calculate the quantity for multiple
    statistics. Returns a complex number with real part corresponding to the mean of real part of the
    visibilities and imaginary part corresponding to the mean of imaginary part of the visibilities """

    mean = {}
    for stats in quality_stats.stats_type:
        stat_obj = getattr(quality_stats, stats)
        Sum = stat_obj['Sum']
        Count = stat_obj['Count']
        mean[stats] = Sum / Count.real

    return mean


def get_variance(quality_stats):
    """ Calculate variance from quality statistics. Returns a dictionary of complex valued arrays with
    real part corresponding to the variance of real part of the visibilities and imaginary part corresponding
    to the variance of imaginary part of the visibilities.
    :return: dictionary {
    'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols), 'freq_stats': ndarray(nfreqs,npols)} """

    variance = {}
    for stats in quality_stats.stats_type:
        stat_obj = getattr(quality_stats, stats)
        SumP2 = stat_obj['SumP2']
        Sum = stat_obj['Sum']
        Count = stat_obj['Count']
        variance[stats] = (SumP2.real - (Sum.real * Sum.real / Count.real)) / (Count.real - 1) + 1.J * (
                SumP2.imag - (Sum.imag * Sum.imag / Count.real)) / (Count.real - 1)

    return variance


def get_differential_mean(quality_stats):
    """ Calculate differential mean from quality statistics. Returns a dictionary of complex valued arrays with
    real part corresponding to the differential mean of real part of the visibilities and imaginary part
    corresponding to the differential mean of imaginary part of the visibilities.
    :return: dict {
    'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols), 'freq_stats': ndarray(nfreqs,npols)} """

    mean = {}
    for stats in quality_stats.stats_type:
        stat_obj = getattr(quality_stats, stats)
        DSum = stat_obj['DSum']
        DCount = stat_obj['DCount']
        mean[stats] = DSum / DCount.real

    return mean


def get_differential_variance(quality_stats):
    """ Calculate differential variance from quality statistics. Returns a dictionary of complex valued arrays with
    real part corresponding to the differential variance of real part of the visibilities and imaginary part
    corresponding to the differential variance of imaginary part of the visibilities.
    :return: dictionary {
    'bl_stats':ndarray(nbls,npols),'time_stats':ndarray(ntimes,npols), 'freq_stats': ndarray(nfreqs,npols)} """

    variance = {}
    for stats in quality_stats.stats_type:
        stat_obj = getattr(quality_stats, stats)
        DSumP2 = stat_obj['DSumP2']
        DSum = stat_obj['DSum']
        DCount = stat_obj['DCount']
        variance[stats] = (DSumP2.real - (DSum.real * DSum.real / DCount.real)) / (DCount.real - 1) + 1.J * (
                DSumP2.imag - (DSum.imag * DSum.imag / DCount.real)) / (DCount.real - 1)

    return variance


def save_moments_to_h5(quality_stats, filename=None, overwrite=False):
    """Save the moments {rfi_percentage, mean, variance, differential mean, differential variance} to a h5 file."""

    if filename is None:
        raise ValueError('Please provide a h5 file')
    elif overwrite:
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

    rfi_percent = get_rfi_percentage(quality_stats)
    mean = get_mean(quality_stats)
    variance = get_variance(quality_stats)
    Dmean = get_differential_mean(quality_stats)
    Dvariance = get_differential_variance(quality_stats)

    moments_list = [rfi_percent, mean, variance, Dmean, Dvariance]

    for i, moment in enumerate(tqdm(moments_list, desc='Writing moments to file: {}'.format(filename))):
        bl_stats_group.create_dataset(moments_name[i], data=moment[quality_stats.stats_type[0]])
        time_stats_group.create_dataset(moments_name[i], data=moment[quality_stats.stats_type[1]])
        freq_stats_group.create_dataset(moments_name[i], data=moment[quality_stats.stats_type[2]])

    print("Moments written to: {}".format(filename))


def load_moments_from_h5(filename=None):
    if filename is None:
        raise ValueError('Please provide a h5 file')

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

    return moments_name, time_moments, freq_moments, ant1, ant2, times, freqs
