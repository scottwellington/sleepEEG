###### Welcome to Group 3's project:

# Deep Learning for Automatic Classification of Sleep Stages

---

![zzz](zzz.png)

##### Preamble:

The dataset is obtainable here: [Sleep-EDF Database](https://www.physionet.org/content/sleep-edfx/1.0.0/)

The full datset is >8.1gb. Naturally, we don't expect you to download all this for marking purposes (and the preprocessing for each data file can take up to a minute), so I have randomly selected test data (as below), keeping the full package to download <50mb.

With apologies, brain-signal processing requires some libraries: I have included a `requirement.txt` to help make this process easier.

This model has been designed to work on the University of Bath's Hex compute cluster; I understand that all testing is to be done on this service, so this code should run without any issues.

I used the script's own randomisation process to select a testing file (participant '4811'). I've not run this test, so I hope you get some interesting results! Our model operates around a typical accuracy of ~70%.

`.py` and `.ipynb` files are provided as requested deliverables (as per instructions). However, **we recommend you clone the repo** onto the Hex compute cluster from:

https://github.com/scottwellington/sleepEEG/

This will provide you with all scripts, models, and test data (including this report!) supplied within the expected directory structure.



***Thank you for your time: we hope you find our project interesting! We certainly did!***

##### Run the .py script as follows:

The difference between whether you are training or testing the model is simply the absence/presence of the `--train` or `--test` flag, so:  

###### To test on the randomly selected participant's data, please run the following command:

```python3 ML2_CW2.py -b model -m CNN --test --in_channels "(6,12,24)" --out_channels "(12,24,48)" --kernel_size "(1,6)" --pool_size "(1,2)" --stride 2 --dropout .2 --padding "(0,0)" --id "4811"```

###### To train on the randomly selected participant's data, please run the following command:

```python3 ML2_CW2.py -b model -m CNN --train --in_channels "(6,12,24)" --out_channels "(12,24,48)" --kernel_size "(1,6)" --pool_size "(1,2)" --stride 2 --dropout .2 --padding "(0,0)" --id "4811"```
