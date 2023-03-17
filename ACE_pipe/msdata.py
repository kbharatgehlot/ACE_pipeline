# noinspection PyPackageRequirements
import os
import numpy as np
import casacore.tables as tbl
from tqdm import tqdm, trange
import h5py
import warnings


class MSData(object):
    """class to read/store/manipulate Measurement set data.
    Attributes
    ----------
        msfile : str
            name of the measurement set loaded to the object.

        ntimes: int
            number of times in the data.

        nfreqs: int
            number of frequency channels in the data.

        nbls: int
            number of unique baselines (including auto-correlations).

        nants: int
            number of antennas.

        times : 1d array
            array of times, Modified Julian Date in units of seconds.

        freqs : 1d array
            array of channel frequencies, in units of Hz.

        uvw : ndarray with shape (ntimes,nbls)
            baseline coordinates for each integration time.

        ant1 : 1d array
            antenna1 for a given baseline.

        ant2 : 1d array
            antenna2 for a given baseline.

        flags : ndarray (ntimes,nbls,nfreqs,npols)
            flag array, default is None.

        data : ndarray (ntimes,nbls,nfreqs,npols)
            data array, default is None.

        weights : ndarray (ntimes,nbls,nfreqs,npols)
            weights array, default is None.
    """

    def __init__(self, filename, uvw, times, freqs, ant1, ant2, ntimes, nfreqs, nbls, nants, columns, flags=None,
                 data=None, weights=None):

        """Create a new MSData object, with default attributes"""

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
        self.columns = columns
        self.flags = flags
        self.data = data
        self.weights = weights

    @staticmethod
    def load(msname, read_flags=False, read_weights=False, read_default_data=False):
        """Function to load the measurement set, metadata, and optionally read FLAG, WEIGHT_SPECTRUM, or DATA columns.
           The read_datacolumn method should be used to read other datacolumns in the MS.

           parameters
           ----------
           msname : str
              name of the MS to load.
           read_flags : bool (Default: False)
              read FLAG column into the flags attribute.

           read_weights : bool (Default: False)
              read WEIGHT_SPECTRUM column into the weights attribute.

           read_default_data : bool (Default : False)
              read DATA column into the data attribute.

            :return: MSData(msname, uvw, times, freqs, ant1, ant2, ntimes, nfreqs, nbls, nants) """

        if not isinstance(msname, str):
            raise ValueError("msname must be a string.")
        elif not os.path.exists(msname):
            raise FileNotFoundError('MS not found in the provided path.')

        with tbl.table(msname, readonly=True) as msobj:
            columns = msobj.colnames()
            times = np.unique(msobj.getcol('TIME'))
            ntimes = len(times)

            with tbl.table(msobj.getkeyword('SPECTRAL_WINDOW'), readonly=True, ack=False) as spw:
                freqs = spw.getcol('CHAN_FREQ').squeeze()
                nfreqs = len(freqs)

            uvw = msobj.getcol('UVW').reshape(ntimes, -1)
            nbls = uvw.shape[1]

            ant1 = (msobj.getcol('ANTENNA1')).reshape(ntimes, -1)[0]
            ant2 = (msobj.getcol('ANTENNA2')).reshape(ntimes, -1)[0]
            nants = len(np.unique(ant1))

        msdata_obj = MSData(msname, uvw, times, freqs, ant1, ant2, ntimes, nfreqs, nbls, nants, columns)

        if read_flags:
            msdata_obj.read_flags()
        if read_default_data:
            msdata_obj.read_datacolumn(read_weights=read_weights)

        return msdata_obj

    def read_datacolumn(self, columnname='DATA', attribute_name='data', read_weights=False):
        """This method reads a given datacolumn in an MS and store it under the attribute_name in MSData object.

           parameters
           ----------
           columnname : str (Default: 'DATA')
              name of the datacolumn to read.

           attribute_name : str (Default: 'data')
              name of the attribute to store the datacolumn.

           read_weights : bool (Default: False)
              read WEIGHT_SPECTRUM column into the weights attribute """

        if getattr(self, attribute_name) is not None:
            raise Exception("MSData.{} already exists".format(attribute_name))

        with tbl.table(self.msfile, readonly=True) as msobj:

            if read_weights:
                if getattr(self, 'weights') is not None:
                    warnings.warn('read_weights=True but weight column is already present. \n'
                                  ' Skipping reading weights', UserWarning)
                else:
                    weights = np.zeros((self.ntimes, self.nbls, self.nfreqs, 4),
                                       dtype=msobj.coldatatype('WEIGHT_SPECTRUM'))

                    for i in trange(self.ntimes, desc='Reading weight spectrum', unit=' integrations',
                                    total=self.ntimes):
                        weights[i] = msobj.getcol('WEIGHT_SPECTRUM', startrow=i * self.nbls, nrow=self.nbls)

                    self.weights = weights

            if columnname not in self.columns:
                raise ValueError("{} does not exist in the MS.".format(columnname))
            else:
                data = np.zeros((self.ntimes, self.nbls, self.nfreqs, 4), dtype=msobj.coldatatype(columnname))
                for i in trange(self.ntimes, desc='Reading column: {}'.format(columnname), unit=' integrations',
                                total=self.ntimes):
                    data[i] = msobj.getcol(columnname, startrow=i * self.nbls, nrow=self.nbls)

        setattr(MSData, attribute_name, data)

    def write_datacolumn(self, columnname=None, data=None, msdata_attr=None):
        """This method writes input data to a given datacolumn in an MS. The input data can be provided using the
        data option of the method or using the msdata_attr inside a MSData object. Only one option can be specified
        at a time. Note: this method only writes to the pre-existing datacolumn in the MS. Functionality to create a
        datacolumn will be added in the future.

            parameters
            ----------
            columnname : str (Default: None)
                name of the datacolumn to write to.

            data : numpy ndarray (Default: None)
                ndarray containing data and with same dimensions as the datacolumn in the MS.

            msdata_attr : str (Default: None)
                MSData attribute name of data column"""

        if data is None:
            if getattr(self, msdata_attr) is None:
                raise ValueError("data array or msdata_attr not provided")
            elif getattr(self, msdata_attr, True):
                raise AttributeError("{} does not exist in the MSData object.".format(msdata_attr))
            else:
                data_to_write = getattr(self, msdata_attr)
        else:
            if getattr(self, msdata_attr) is not None:
                raise Exception("Both data and msdata_attr provided.")
            else:
                data_to_write = data

        with tbl.table(self.msfile, readonly=False) as msobj:
            for i in trange(self.ntimes, desc='Writing data to {}'.format(columnname), unit=' integrations',
                            total=self.ntimes):
                msobj.putcol(columnname, data_to_write[i], startrow=i * self.nbls, nrow=self.nbls)
            print("Data written")

    def read_flags(self, overwrite=False):
        """This method reads the default flag column of the MS to flags attribute of MSData object.

            parameters
            ----------
            overwrite : bool (Default: False)
                overwrite pre-existing flags in MSData object if set to True."""

        if self.flags is not None and overwrite is False:
            raise Exception("The flags already exist in the MSData object. Set overwrite=True to overwrite the "
                            "current flags")
        else:
            if overwrite:
                warnings.warn("overwrite=True, flags will be overwritten", UserWarning)

            flags = np.zeros((self.ntimes, self.nbls, self.nfreqs, 4), dtype=np.int8)
            with tbl.table(self.msfile, readonly=True) as msobj:
                for i in trange(self.ntimes, desc='Reading Flags', unit=' integrations', total=self.ntimes):
                    flags[i] = msobj.getcol('FLAG', startrow=i * self.nbls, nrow=self.nbls)
            self.flags = flags

    def write_flags(self, flags=None, use_msdata_flags=False):
        """This method writes input flags or flags attribute of the MSData object to the default flag column of the MS.

            parameters
            ----------
                flags : numpy ndarray (Default: None)
                    ndarray containing flags and with same dimensions as the flagcolumn in the MS.
                use_msdata_flags : bool (Default: False)
                    write the flags attribute of the MSData object instead of the input flags"""
        if flags is None and not use_msdata_flags:
            raise ValueError("Either provide input flags or use_msdata_flags.")
        elif flags is not None and use_msdata_flags:
            raise Exception("Both flags and use_msdata_flags provided. Only one option is allowed at a time.")
        elif flags is None and getattr(self, 'flags') is None:
            raise AttributeError("flags attribute in the MSData object is empty.")
        else:
            if use_msdata_flags:
                flags = self.flags
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
        """Create a new QualityStatistics object.
        Attributes
        ----------
        stats_name : dict (str)
            dictionary with the statistic names
        stats_kind : array (int)
            index associated with stats name
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
        """Function to load the QualityStatistics from the measurement set.
            parameters
            ----------
            msname : str
                name of the MS.
        """
        if not os.path.exists(msname):
            raise FileNotFoundError('MS not found in the provided path.')
        elif not os.path.exists(msname + "/QUALITY_KIND_NAME"):
            raise FileNotFoundError('Quality statistics not found in the MS.')

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
