from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os

# MAKE SURE THESE VARIABLES HAVE BEEN SET
os.environ['COMET_API_KEY'] = 'TKbzjJtemM52RdLukBpprqCNv'
os.environ['MY_PROJECT_NAME'] = 'yq2240'

assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
print("Running experiment for project: {}".format(os.environ['MY_PROJECT_NAME']))

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'

from comet_ml import Experiment


class BayesridgeFlow(FlowSpec):
    """
    SampleRegressionFlow is a minimal DAG showcasing reading data from a file 
    and training a model successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text file with the dataset',
        is_text=True,
        default='regression_dataset.txt')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )

    VALIDATION_SPLIT = Parameter(
        name='validation_split',
        help='validation dataset size',
        default=0.20
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        from io import StringIO

        raw_data = StringIO(self.DATA_FILE).readlines()
        print("Total of {} rows in the dataset!".format(len(raw_data)))
        self.dataset = [[float(_) for _ in d.strip().split('\t')] for d in raw_data]
        print("Raw data: {}, cleaned data: {}".format(raw_data[0].strip(), self.dataset[0]))
        self.Xs = [[_[0]] for _ in self.dataset]
        self.Ys =  [_[1] for _ in self.dataset]
        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        assert(all(y < 100 and y > -100 for y in self.Ys))
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xs, 
            self.Ys, 
            test_size=self.TEST_SPLIT, 
            random_state=42
            )
        self.x_train,self.x_vali,self.y_train,self.y_vali=train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=self.VALIDATION_SPLIT / (1 - self.TEST_SPLIT), 
            random_state=42
            )

        self.next(self.hyperparameters)

    @step
    def hyperparameters(self):

       self.N_iter=[100,200,300,400]
       self.next(self.train_model,foreach="N_iter")

    @step
    def train_model(self):
        """
        Train a regression on the training set
        """
        from sklearn import linear_model
        self.n_iter=self.input
        reg = linear_model.BayesianRidge(n_iter=self.input)
        reg.fit(self.x_train, self.y_train)
        print("Coefficient {}, intercept {}".format(reg.coef_[0], reg.intercept_))
        # now, make sure the model is available downstream
        self.model = reg
        # go to the testing phase
        self.next(self.validation)



    @step 
    def validation(self,inputs):
        """
        Test the model on the hold out sample
        """
        from sklearn import metrics
        import numpy as np
        #self.exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'],
                    #auto_param_logging=False)
        self.mses=[]
        self.r_2s=[]
        self.models=[input.model for input in inputs]
        self.n_iters=[input.n_iter for input in inputs]
        self.merge_artifacts(inputs, include=['x_train','y_train','x_vali','y_vali','X_test','y_test'])
        for input in inputs:
            self.y_predicted = input.model.predict(input.x_vali)
            self.mse = metrics.mean_squared_error(input.y_vali, self.y_predicted)
            self.r2 = metrics.r2_score(input.y_vali, self.y_predicted)
            self.mses.append(self.mse)
            self.r_2s.append(self.r2)
        

        for i in range(len(self.mses)):
            self.mse=self.mses[i]
            self.r2=self.r_2s[i]
            self.n_iter=self.n_iters[i]
            self.model=self.models[i]

        self.idx = np.argmin(self.mses)
        self.mse=self.mses[self.idx]
        self.r2=self.r_2s[self.idx]
        self.n_iter=self.n_iters[self.idx]
        self.model=self.models[self.idx]
        print('The best model based on MSE has MSE : {}, R2 score : {}'.format(self.mse,self.r2))

        all_validation_statistics = {"n_iter_1": self.n_iters[0],
                                          "n_iter_2": self.n_iters[1],
                                          "n_iter_3": self.n_iters[2],
                                          "n_iter_4": self.n_iters[3],
                                          "mse_1": self.mses[0],
                                          "mse_2": self.mses[1],
                                          "mse_3": self.mses[2],
                                          "mse_4": self.mses[3],
                                          "r2_1": self.r_2s[0],
                                          "r2_1": self.r_2s[1],
                                          "r2_1": self.r_2s[2],
                                          "r2_1": self.r_2s[3],                                     
                                      "Best_niter": self.n_iter,
                                      "Best_r2":self.r2,
                                      "Best_mse":self.mse}
        exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'], auto_param_logging=False)
        exp.log_dataset_hash(self.x_train)
        
        
        exp.log_parameters({"N_iter": self.n_iter})
        exp.log_metrics(all_validation_statistics)
        
        #self.r2 = metrics.r2_score(self.y_test, self.y_predicted)

        #print('MSE is {}, R2 score is {}'.format(self.mse, self.r2))
        # print out a test prediction
        #test_predictions = self.model.predict([[10]])
        #print("Test prediction is {}".format(test_predictions))
        # all is done go to the end
        self.next(self.end)

    

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    BayesridgeFlow()
