#################
### Preamble: ###
#################


The full datset is >8.1gb. We don't expect you to download all this for marking purposes, so I have randomly selected test data (as below).
With apologies, brain-signal processing requires some libraries: you may need to install some packages (cf. .py script imports).

This model has been designed to work on the University of Bath's Hex compute cluster; I understand that all testing is to
be done on this service, so this code should run without any issues.

As I'm writing this somewhat at the eleventh hour, I've randomly selected participant data "4241" from the dataset for testing purposes.
I've not run this test, so I hope you get some interesting results! Our model operates around a typical accuracy of ~70%.

.py and .ipynb files are provided as requested deliverables (as per instructions). However, WE RECOMMEND YOU CLONE THE REPO onto the Hex compute cluster from:

https://github.com/scottwellington/sleepEEG/

This will provide you with all scripts, models, and test data (including this report!) supplied within the expected directory structure.



Thank you for your time: we hope you find our project interesting! We certainly did!


######################################
### Run the .py script as follows: ###
######################################


# To test on the randomly selected participant's data, please run the following command:

# python3 ML2_CW2.py -b model -m CNN --test --in_channels "(6,12,24)" --out_channels "(12,24,48)" --kernel_size "(1,6)" --pool_size "(1,2)" --stride 2 --dropout .2 --padding "(0,0)" --id "4241"



# To train on the randomly selected participant's data, please run the following command:

# python3 ML2_CW2.py -b model -m CNN --train --in_channels "(6,12,24)" --out_channels "(12,24,48)" --kernel_size "(1,6)" --pool_size "(1,2)" --stride 2 --dropout .2 --padding "(0,0)" --id "4241"
