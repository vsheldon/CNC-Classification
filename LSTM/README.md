# LSTM in classifing CNC machine data
to use the script, first install following packages with example command

`sudo pip install numpy tensorflow sklearn matplotlib`

then format your input data in the following convertion:

* input data is a matrix with dimension `m x n`, where `m` is number of measurements, `n` is number of features in each measurement

* input label is a array with dimension `m x 1`, where 'm' is the label for each measurement.

to start training, run:

`python tsc_main.py`

