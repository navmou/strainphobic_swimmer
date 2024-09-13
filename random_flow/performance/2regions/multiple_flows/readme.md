# Performance evaluation

The codes evaluates the perfromance of give policies in their time
spent in favorable region of the flow, and the final vertical migration.

- file names follow the convention {signal}.cpp
- plotting scripts are to plot the rewards and normalized rewards.
- The naive, passive and upward does not have any dependence on state so there is only one file for each of them
- random action taking might result in slightly different values for different state signals therefor there are random
codes for each signal. (8 files.)

## field-read.h 
Loads the flow data.

## submit.h
Is a sample script to submit the jobs in cluster. 
For each signal it submits the code for 11 different $\beta$ values.
(Edit for each signal by putting the correct policies)
