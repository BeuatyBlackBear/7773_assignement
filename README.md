# 7773_assignement
- The website on github is :https://github.com/BeuatyBlackBear/7773_assignement
- The comet link is https://www.comet.com/beuatyblackbear/yq2240/dab73ec6fa1d44669b7b119228d19d62?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=wall.
* `requirements.txt` holds the Python dependencies required for running the scripts.
## Task 1:
* `task1` is the folder containing the scripts for task1. You should first got the path ./task1 first;
- `regression_dataset.txt` is a fake regression dataset to run the code - you can generate another dataset with different parameters or cardinality using the `create_fake_dataset.py` script and use the syntax `python create_fake_dataset.py`;
* `flow.py` contains the script for task one. To run the script, please use the syntax : `python flow.py run`.
*  `metrics.png` contains the screenshot on comet to record all models and  best model's metrics on validation set (MSE and R2).

## Task 2:
* `task2` is the folder containing the scripts for task 2. You should first got the path ./task2 first;
* `flow_2.py` contains the script for task one. To run the script, please use the syntax e.g.  `python flow_2.py run --n_iter '[100,200,300,400]'`.
* Note: `n_iter`is the parameter name, you have to set the parameter in the form of string of list. You can change the number in the list. It is beter to set the parameters to be times of one hunderd.






