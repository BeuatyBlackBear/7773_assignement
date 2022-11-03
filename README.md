# 7773_assignement
## Task 1:
* `7773-assignment` is the folder containing the scripts we will described during the class;
* `requirements.txt` holds the Python dependencies required for running the scripts.
- `regression_dataset.txt` is a fake regression dataset to run the code - you can generate another dataset with different parameters or cardinality using the `create_fake_dataset.py` script and use the syntax `python create_fake_dataset.py`;
* `flow.py` contains the script for task one. To run the script, please use the syntax : `COMET_API_KEY=xxx MY_PROJECT_NAME=yyy python flow.py run` with the API key.
*  `hyperparamters` contains the screenshot on comet to record the best model's parameter.
*  `metrics` contains the screenshot on comet to record the best model's metrics on test set (MSE and R2).

## Task 2:

* `flow_2.py` contains the script for task one. To run the script, please use the syntax e.g.  `COMET_API_KEY=xxx MY_PROJECT_NAME=yyy python flow_2.py run --n_iter '[100,200,300,400]'` with the API key. 
* Note: `n_iter`is the parameter name, you have to set the parameter in the form of string of list. You can change the number in the list. It is beter to set the parameters to be times of one hunderd.



