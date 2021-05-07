#################
### Preamble: ###
#################


The full datset is >8.1gb. We don't expect you to download all this for marking purposes, so I have randomly selected test data (as below).
With apolgies, brain-signal processing requires some libraries: You may need to install some packages (all at top of script).

This model has been designed to work on the University of Bath's HEX compute cluster; I understand that all testing is to
be done on this service, so this code should run without any issues.

As I'm writing this at the eleventh hour, I've randomly selected participant data "4241" from the dataset for testing purposes.
I've not run this test, so I hope you get some interesting results! Our model operates around a typical accuracy of ~70%.



###################################
### Run this script as follows: ###
###################################


To test on the randomly selected participant's data, please run the following command:

python3 ML2_CW2.py -b model -m CNN --test --in_channels "(6,12,24)" --out_channels "(12,24,48)" --kernel_size "(1,6)" --pool_size "(1,2)" --stride 2 --dropout .2 --padding "(0,0)" --id "4241"



To train on the randomly selected participant's data, please run the following command:

python3 ML2_CW2.py -b model -m CNN --train --in_channels "(6,12,24)" --out_channels "(12,24,48)" --kernel_size "(1,6)" --pool_size "(1,2)" --stride 2 --dropout .2 --padding "(0,0)" --id "4241"
