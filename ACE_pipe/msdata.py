# noinspection PyPackageRequirements
import numpy as np
import casacore.tables as tbl
from tqdm import tqdm, trange
import h5py


class MSData(object):

    def __init__(self, filename, uvw, times, freqs, ant1, ant2, ntimes, nfreqs, nbls, nants, flags=None, data=None,
                 weights=None):
        self.msfile = filename
        self.uvw = uvw
        self.times = times
        self.freqs = freqs
        self.ant1 = ant1
        self.ant2 = ant2

        self.ntimes = ntimes
        self.nfreqs = nfreqs
        self.nbls = nbls
        self.nants = nants
        self.flags = flags
        self.data = data
        self.weights = weights

    @staticmethod
    def load(msname, read_flags=False, read_weights=False, read_default_data=False):
        """Function to load the measurement set and metadata. It does not read any data columns. For that purpose,
        you may want to use the read_datacolumn method """
        if msname is None:
            raise ValueError('This is cheating, you gotta provide a measurement set in order to load it.')
        # if os._exists(msname) is False:
        #    raise FileNotFoundError('MS not found in the provided path.')
        with tbl.table(msname, readonly=True) as msobj:
            times = msobj.getcol('TIME')
            ntimes = len(np.unique(times))

            with tbl.table(msobj.getkeyword('SPECTRAL_WINDOW'), readonly=True, ack=False) as spw:
                freqs = spw.getcol('CHAN_FREQ').squeeze()
                nfreqs = len(freqs)

            uvw = msobj.getcol('UVW')
            nbls = int(uvw.shape[0] / ntimes)

            ant1 = (msobj.getcol('ANTENNA1')).reshape(ntimes, -1)[0]
            ant2 = (msobj.getcol('ANTENNA2')).reshape(ntimes, -1)[0]
            nants = len(np.unique(ant1))

        msdata_obj = MSData(msname, uvw, times, freqs, ant1, ant2, ntimes, nfreqs, nbls, nants)

        if read_flags:
            msdata_obj.read_flags()
        if read_default_data:
            msdata_obj.read_datacolumn(read_weights=read_weights)

        return msdata_obj

    def read_datacolumn(self, columnname='DATA', attribute_name='data', read_weights=False):
        """This method reads a given datacolumn in an MS and save it under the attribute name provided. The default
        datacolumn is 'DATA' and attribute name is 'data' """

        with tbl.table(self.msfile, readonly=True) as msobj:
            if read_weights and self.weights is None:
                weights = np.zeros((self.ntimes, self.nbls, self.nfreqs, 4), dtype=float)
                for i in trange(self.ntimes, desc='Reading weight spectrum', unit=' integrations', total=self.ntimes):
                    weights[i] = msobj.getcol('WEIGHT_SPECTRUM', startrow=i * self.nbls, nrow=self.nbls)
                self.weights = weights
            else:
                print('Weight spectrum already present')
                pass

            data = np.zeros((self.ntimes, self.nbls, self.nfreqs, 4), dtype=np.complex_)
            for i in trange(self.ntimes, desc='Reading column: {}'.format(columnname), unit=' integrations',
                            total=self.ntimes):
                data[i] = msobj.getcol(columnname, startrow=i * self.nbls, nrow=self.nbls)

        if columnname != 'DATA' or attribute_name != 'data':
            setattr(MSData, attribute_name, data)
        else:
            self.data = data

    def write_datacolumn(self, columnname='DATA', data=None):
        if data is None:
            raise ValueError("data array cannot be None.")
        else:
            with tbl.table(self.msfile, readonly=False) as msobj:
                for i in trange(self.ntimes, desc='Writing data to {}'.format(columnname), unit=' integrations',
                                total=self.ntimes):
                    msobj.putcol(columnname, data[i], startrow=i * self.nbls, nrow=self.nbls)
            print("Data written")

    def read_flags(self):
        """This method reads the default flag column of an MS"""
        flags = np.zeros((self.ntimes, self.nbls, self.nfreqs, 4), dtype=np.int8)
        with tbl.table(self.msfile, readonly=True) as msobj:
            for i in trange(self.ntimes, desc='Reading Flags', unit=' integrations', total=self.ntimes):
                flags[i] = msobj.getcol('FLAG', startrow=i * self.nbls, nrow=self.nbls)
        self.flags = flags

    def write_flags(self, flags=None):
        if flags is None:
            raise ValueError("flags cannot be None.")
        else:
            with tbl.table(self.msfile, readonly=False) as msobj:
                for i in trange(self.ntimes, desc='Writing Flags', unit=' integrations', total=self.ntimes):
                    msobj.putcol('FLAG', flags[i], startrow=i * self.nbls, nrow=self.nbls)
            print("Flags written")


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
    sumMeanSquared = sum * sum / n;
    return (sumP2 - sumMeanSquared) / (n - 1.0);
    """

    def __init__(self, stats_name, stats_kind, stats_type, bl_stats, time_stats, freq_stats, ant1, ant2, times, freqs):
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

        stats_name = {}
        with tbl.table(msname + "/QUALITY_KIND_NAME", readonly=True) as statobj:
            stats_kind = statobj.getcol('KIND')
            for i, kind in enumerate(stats_kind):
                stats_name[kind] = statobj.getcol('NAME', startrow=i, nrow=1)[0]

        print("Reading baseline quality statistics...")
        bl_stats = {}
        with tbl.table(msname + "/QUALITY_BASELINE_STATISTIC", readonly=True) as statobj:
            ant1 = statobj.getcol('ANTENNA1').reshape(-1, len(stats_kind))[:, 0]
            ant2 = statobj.getcol('ANTENNA2').reshape(-1, len(stats_kind))[:, 0]
            values = statobj.getcol('VALUE').reshape(-1, len(stats_kind), 4)
            for i, kind in enumerate(stats_kind):
                bl_stats[stats_name[kind]] = values[:, i, :]

        print("Reading frequency quality statistics...")
        freq_stats = {}
        with tbl.table(msname + "/QUALITY_FREQUENCY_STATISTIC", readonly=True) as statobj:
            freqs = np.unique(statobj.getcol('FREQUENCY'))
            values = statobj.getcol('VALUE').reshape(-1, len(stats_kind), 4)
            for i, kind in enumerate(stats_kind):
                freq_stats[stats_name[kind]] = values[:, i, :]

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
        if filename is None:
            raise ValueError('Please provide a h5 file')

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

            for i, kind in enumerate(tqdm(stats_kind, desc='Reading quality statistics from file:{}'.format(filename))):
                stats_name[kind] = name[i]
                bl_stats[name[i]] = bl_stat_table[name[i]][:]
                time_stats[name[i]] = time_stat_table[name[i]][:]
                freq_stats[name[i]] = freq_stat_table[name[i]][:]

        return QualityStatistics(stats_name, stats_kind, stats_type, bl_stats, time_stats, freq_stats, ant1, ant2,
                                 times, freqs)

    def save_statistics_to_h5(self, filename=None, overwrite=False):
        """Save the quality statistics to a h5 file."""
        if filename is None:
            raise ValueError('Please provide a h5 file')
        elif overwrite:
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
