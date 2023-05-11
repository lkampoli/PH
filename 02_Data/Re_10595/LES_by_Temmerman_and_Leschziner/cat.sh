cat data-001.dat data-003.dat data-004.dat data-005.dat data-006.dat data-007.dat data-008.dat data-009.dat data-010.dat > all_data.dat
# replace spaces with commas ...
cut -d "," -f2 all_data.dat > U0.csv
cut -d "," -f3 all_data.dat > V0.csv
cut -d "," -f7 all_data.dat > K0.csv
