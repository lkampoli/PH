#!/usr/bin/env bash

for run1 in running_pfac*
do
  cd $run1
  for run2 in `ls -d run_*[0-9]`
  do
    echo $run2
    zip -r -q $run2.zip $run2
    rm -r $run2
  done
  cd ..
done
