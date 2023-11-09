# noinspection PyPackageRequirements
import os
import numpy as np
import casacore.tables as tbl
from tqdm import tqdm, trange
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

        npols: int
            number of correlations (polarizations) in the data.
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

    """

    def __init__(self, filename, uvw, times, freqs, ant1, ant2, ntimes, nfreqs, npols, nbls, nants, columns):

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
        self.npols = npols
        self.nants = nants
        self.columns = columns

    @staticmethod
    def load(msname, read_flags=False, read_weights=False, read_default_data=False):
        """Function to load the measurement set, metadata, and optionally read FLAG, WEIGHT_SPECTRUM, or DATA columns.
           The read_datacolumn method should be used to read other data columns in the MS.

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

        assert isinstance(msname, str), "msname must be a string."

        try:
            assert os.path.exists(msname)

            with tbl.table(msname, readonly=True) as msobj:
                columns = msobj.colnames()
                times = np.unique(msobj.getcol('TIME'))
                ntimes = len(times)

                with tbl.table(msobj.getkeyword('SPECTRAL_WINDOW'), readonly=True, ack=False) as spw:
                    freqs = spw.getcol('CHAN_FREQ').squeeze()
                    nfreqs = len(freqs)

                with tbl.table(msobj.getkeyword('POLARIZATION'), readonly=True, ack=False) as pol:
                    npols = pol.getcol('NUM_CORR').squeeze()

                uvw = msobj.getcol('UVW').reshape(ntimes, -1, 3)
                nbls = uvw.shape[1]

                ant1 = (msobj.getcol('ANTENNA1')).reshape(ntimes, -1)[0]
                ant2 = (msobj.getcol('ANTENNA2')).reshape(ntimes, -1)[0]
                nants = len(np.unique(ant1))

                msdata_obj = MSData(msname, uvw, times, freqs, ant1, ant2, ntimes, nfreqs, npols, nbls, nants, columns)

                if read_flags:
                    msdata_obj.read_flags()
                if read_default_data:
                    msdata_obj.read_datacolumn(read_weights=read_weights)

            return msdata_obj
        except FileNotFoundError:
            print("MS not found in the provided path.")
            raise

    def read_datacolumn(self, columnname='DATA', attribute_name='data', overwrite=False, read_weights=False):
        """This method reads a given datacolumn in an MS and store it under the attribute_name in MSData object.

            parameters
            ----------
            columnname : str (Default: 'DATA')
                name of the datacolumn to read.

            attribute_name : str (Default: 'data')
                name of the attribute to store the datacolumn.

            overwrite: bool (Default: False)
                overwrite the present msdata attribute.

            read_weights : bool (Default: False)
                read WEIGHT_SPECTRUM column into the weights attribute """

        if hasattr(self, attribute_name):
            if overwrite:
                warnings.warn('MSData.{} already exists and but will be overwritten.', UserWarning)
            else:
                raise Exception("MSData.{} already exists and overwrite is set to False.".format(attribute_name))
        else:
            pass

        with tbl.table(self.msfile, readonly=True) as msobj:

            if read_weights:
                if hasattr(self, 'weights'):
                    warnings.warn('read_weights=True but weight column is already present in MSData. \n'
                                  ' Skipping reading weights', UserWarning)
                else:
                    weights = np.zeros((self.ntimes, self.nbls, self.nfreqs, 4),
                                       dtype=msobj.coldatatype('WEIGHT_SPECTRUM'))

                    for i in trange(self.ntimes, desc='Reading weight spectrum', unit=' integrations',
                                    total=self.ntimes):
                        weights[i] = msobj.getcol('WEIGHT_SPECTRUM', startrow=i * self.nbls, nrow=self.nbls)

                    setattr(MSData, 'weights', weights)

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
            if hasattr(self, msdata_attr):
                data_to_write = getattr(self, msdata_attr)
            else:
                raise ValueError("Input data array not provided and msdata_attr does not exist")
        else:
            if hasattr(self, msdata_attr):
                warnings.warn('Both data and msdata_attr are provided. Input data will be used.', UserWarning)
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

        if hasattr(self, 'flags'):
            if overwrite:
                warnings.warn("overwrite=True, flags will be overwritten", UserWarning)
            else:
                raise Exception(
                    "The flags already exist in the MSData object. Set overwrite=True to overwrite the current flags")
        else:
            pass

        flags = np.zeros((self.ntimes, self.nbls, self.nfreqs, 4), dtype=np.int8)
        with tbl.table(self.msfile, readonly=True) as msobj:
            for i in trange(self.ntimes, desc='Reading Flags', unit=' integrations', total=self.ntimes):
                flags[i] = msobj.getcol('FLAG', startrow=i * self.nbls, nrow=self.nbls)
            setattr(self, 'flags', flags)

    def write_flags(self, flags=None, use_msdata_flags=False):
        """This method writes input flags or flags attribute of the MSData object to the default flag column of the MS.

            parameters
            ----------
                flags : numpy ndarray (Default: None)
                    ndarray containing flags and with same dimensions as the flag column in the MS.
                use_msdata_flags : bool (Default: False)
                    write the flags attribute of the MSData object instead of the input flags."""

        if flags is None:
            if use_msdata_flags and hasattr(self, 'flags'):
                flags_to_write = self.flags
            else:
                raise AttributeError(
                    "Either flags attribute in the MSData object is empty or use_msdata_flags is set to False.")
        else:
            if use_msdata_flags:
                warnings.warn("use_msdata_flags also set to True, but will not be used", UserWarning)
            flags_to_write = flags

        with tbl.table(self.msfile, readonly=False) as msobj:
            for i in trange(self.ntimes, desc='Writing Flags', unit=' integrations', total=self.ntimes):
                msobj.putcol('FLAG', flags_to_write[i], startrow=i * self.nbls, nrow=self.nbls)
            print("Flags written")