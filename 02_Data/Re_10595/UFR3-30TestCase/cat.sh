cat UFR3-30_C_10595_data_MB-001.dat \
    UFR3-30_C_10595_data_MB-002.dat \
    UFR3-30_C_10595_data_MB-003.dat \
    UFR3-30_C_10595_data_MB-004.dat \
    UFR3-30_C_10595_data_MB-005.dat \
    UFR3-30_C_10595_data_MB-006.dat \
    UFR3-30_C_10595_data_MB-007.dat \
    UFR3-30_C_10595_data_MB-008.dat \
    UFR3-30_C_10595_data_MB-009.dat \
    UFR3-30_C_10595_data_MB-010.dat  > UFR_data.csv

# replace spaces with commas ...
#cut -d "," -f2 UFR_data.csv > U0.csv
#cut -d "," -f3 UFR_data.csv > V0.csv
#cut -d "," -f7 UFR_data.csv > K0.csv

#cp U0.csv U0.edf
#cp V0.csv V0.edf
#cp K0.csv K0.edf

#gzip U0.edf
#gzip V0.edf
#gzip K0.edf
