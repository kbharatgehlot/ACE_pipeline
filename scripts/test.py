import numpy as np
from ace_pipe.msdata import MSData
from ace_pipe.qualitystatistics import QualityStatistics

def main():
    filename='/home/bharat/DP3_test/ACE_SB371-384_20sec-42ch4s_LST23h30_08.MS'
    ms_data = MSData.load(filename,read_flags=True,read_weights=True,read_default_data=True)
    assert ms_data.data.shape == (ms_data.ntimes,ms_data.nbls,ms_data.nfreqs,4)


if __name__ == '__main__':
    main()


    #ms_data.load_datacolumn(columnname='DATA',read_weights=True,attribute_name='mrdoctor')

#print(ms_data.mrdoctor.shape)
#print(dir(ms_data))
#quality_stats = load_from_ms(filename)
#print(quality_stats.stats_name)
#print(quality_stats.bl_stats['RFICount'].shape)
#print(quality_stats.time_stats['RFICount'].shape)
#print(quality_stats.freq_stats['RFICount'].shape)
#quality_stats.save_statistics_to_h5(filename='/home/bharat/DP3_test/ACE_SB371-384_20sec-42ch4s_LST23h30_08.MS/test_stats.h5',overwrite=True)
#quality_stats2 = QualityStatistics.load_from_h5(filename='/home/bharat/DP3_test/ACE_SB371-384_20sec-42ch4s_LST23h30_08.MS/test_stats.h5')

#print(quality_stats.stats_name)
#print(quality_stats2.stats_name)

#stat_obj = getattr(quality_stats2, 'time_stats')

#rfi_percent = utils.get_rfi_percentage(quality_stats)
#print(rfi_percent)

#quality_stats2.save_moments_to_h5(filename='/home/bharat/DP3_test/ACE_SB371-384_20sec-42ch4s_LST23h30_08.MS/test_moments.h5',overwrite=True)

#bad_baselines = flagger.get_bad_baselines(quality_stats2,nsigma_mean=5,nsigma_var=5)
#print(len(bad_baselines))