import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_covtype
from sklearn import naive_bayes as nb
from sklearn import neural_network as nn
from IPython.core.display import display
import time
import codes as c

def read_covtype(engineer=False):
    '''
    Name:
        read_covtype
    
    Purpose: 
        Read covtype dataset using the sklearn.datasets function fetch_covtype 
        and return in X, y array format along with class name and number arrays
    
    Parameters: 
        No Required Inputs:
        
        1 Optional Settings:
                 
        engineer = Boolean, default=False. Use feature engineering pre-processing 
        to add additional features and compress binary features 
    
    Returns: 
        4 Ouputs: 
        
        X = NumPy array, data array
        y = NumPy array, class labels
        cnames = list, class names
        cnums = NumPy array, class number (numeric class labels)
    '''     
    
    if engineer==True:
        data = c.load_covertype_data()
        data_new = c.compress_and_engineer_features(data)
        X, y = c.get_features_and_labels(data_new)
        
    else:
        data = fetch_covtype()
        X = data['data']
        y = data['target']

    cnames = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
    cnums = np.arange(1,8)
    
    return X, y, cnames, cnums
    

def mlp_explore_param(param, values, X_train, y_train, X_test, y_test, args=None, redo=False):
    '''
    Name:
        mlp_explore_param
    
    Purpose: 
        Explore a given scikit-learn MLP hyperparameter by looping through an 
        array of values and recording classification performance for each step. 
        Results are be written to CSV files and read from there on subsequent 
        calls unless redo=True.
    
    Parameters: 
        6 Required Inputs:
        
        param = String, hyperparameter to test (e.g. 'hidden_layer_sizes')
        values = List or numpy array containing param values to be tested
        X_train = NumPy array, Training data
        y_train = NumPy array, Training labels
        X_test = NumPy array, Test data
        y_test = NumPy array, Test labels
        
        2 Optional Settings:
        
        args = Dictionary, default={'solver':'sgd', 'early_stopping':True}. 
               Arguments passed to the MLP classifier via **kwargs that 
               will be kept constant for each test
        redo = Boolean, default=False. Results will be written to and read 
               from a CSV file. Set redo=True to remake an existing CSV
        
    Returns: 
        Out: Pandas DataFrame containing the metrics (accuracy, f1, precision, 
             recall, run time, loss, and iteration count) for each parameter value
    ''' 
    
    #default arguments
    kwargs = args if args != None else {'solver':'sgd', 'early_stopping':True}
    
    #add input parameter to kwargs if necessary
    if param not in kwargs:
        kwargs = {**kwargs, **{param:values[0]}}

    #create blank output arrays    
    runtimes = np.zeros(np.shape(values)[0])
    n_iter = np.zeros(np.shape(values)[0])
    loss = np.zeros(np.shape(values)[0])
    
    #define csv file name for storing results table
    if param=='learning_rate_init' or param=='tol':
        file = Path('mlp_explore_results/mlp_explore_'+param+'_'+kwargs['learning_rate']+'.csv')
    elif param=='hidden_layer_sizes':
        config = 'depth' if type(values)==list else 'width'
        file = Path('mlp_explore_results/mlp_explore_'+param+'_'+config+'.csv')
    else:
        file = Path('mlp_explore_results/mlp_explore_'+param+'.csv')
    
    #if the file exists, read results from there unless redo keyword is set
    if file.is_file()==True and redo==False:
        
        print('Reading existing results from '+file.name)
        df_totals = pd.read_csv(file.resolve(), index_col=0)

    else:
    
        print('Exploring param = '+param+' from '+str(values[0])+' to '+str(values[-1]))
        print('Working on '+param+' = ', sep=' ', end='', flush=True)
    
        #loop through input values for param
        for i, param_value in enumerate(values):
            
            #insert current value into kwargs and print
            kwargs[param] = param_value
            print(param_value, sep=' ', end=',', flush=True)
            
            start = time.time()
            
            #initialise classifier, train, and predict
            _classifier = nn.MLPClassifier(**kwargs)
            _classifier = _classifier.fit(X_train, y_train)
            y_pred = _classifier.predict(X_test)
            
            #store runtime, number of iterations, and final loss function value
            runtimes[i] = time.time()-start
            n_iter[i] = _classifier.n_iter_
            loss[i] = _classifier.loss_
            
            #construct confusion matrix and format metrics into dataframe
            conf = c.construct_confusion_matrix(y_test, y_pred, dim=7)
            df_total, df_class, df_conf = c.metrics_wrapper(conf, cnames, do_display=False)

            #initialise or append to main results output dataframe  
            df_totals = df_total.copy() if i==0 else df_totals.append(df_total, ignore_index=True)

        #insert additional information into output    
        df_totals.insert(0, 'n_iter', n_iter)    
        df_totals.insert(0, 'loss', loss)    
        df_totals.insert(0, 'Run Time', runtimes)
        df_totals.insert(0, param, values)

        #write output dataframe to csv file
        df_totals.to_csv(path_or_buf=file)
        print(' ', sep='\newline')
        print('Wrote results to '+file.name)
    
    return df_totals
    
def mlp_explore_params(X_train, y_train, X_test, y_test, redo=False):
    '''
    Name:
        mlp_explore_params
    
    Purpose: 
        Explore scikit-learn MLP hyperparameters by looping through an 
        arrays possible values. This code is wrapper for mlp_explore_param 
        that defines the the values to be tested and passes them to the main 
        routine. 
    
    Parameters: 
        4 Required Inputs:
    
        X_train = NumPy array, Training data
        y_train = NumPy array, Training labels
        X_test = NumPy array, Test data
        y_test = NumPy array, Test labels
        
        1 Optional Settings:
        
        redo = Boolean, default=False. Results will be written to and read 
               from CSV filee. Set redo=True to remake existing CSVe
        
    
    Returns: 
        Out: List with Pandas DataFrames containing the metrics (accuracy, f1, precision, 
             recall, run time, loss, and iteration count) for each parameter value
    ''' 
    
    #hyperparameters accepted by MLPClassifier to be tested
    params = ['hidden_layer_sizes', \
              'hidden_layer_sizes', \
              'activation', \
              'alpha', \
              'batch_size', \
              'momentum', \
              'learning_rate_init', \
              'shuffle', \
              'nesterovs_momentum', \
              'power_t', \
              'tol']
    
    param_titles = ['Single Hidden Layer (Width)', \
                    'Multiple Hidden Layers (Depth)', \
                    'Activation Function', \
                    'Alpha (L2 Penalty)', \
                    'Mini Batch Size', \
                    'Momentum', \
                    'Initial Learning Rate (Rate = Constant)', \
                    'Initial Learning Rate (Rate = Inverse Scaling)', \
                    'Initial Learning Rate (Rate = Adaptive)', \
                    'Shuffle', \
                    'Nesterov''s Momentum', \
                    'Inverse Scaling Exponent', \
                    'Tolerance (Rate = Constant)', \
                    'Tolerance (Rate = Inverse Scaling)', \
                    'Tolerance (Rate = Adaptive)']    
    
    #test values corresponding to params array above
    values = [np.append([np.arange(1,10,1),np.arange(10,100,10)],np.arange(100,1100,100)), \
              [[100],[100,100],[100,100,100],[100,100,100,100],[100,100,100,100,100]], \
              ['identity', 'logistic', 'tanh', 'relu'], \
              np.sort(np.append(np.geomspace(1e-6,1e-1,num=6),np.geomspace(5e-6,5e-1,num=6))), \
              np.sort(np.append(np.geomspace(1e1,1e5,num=5),np.geomspace(5e1,5e4,num=4))).round().astype(int), \
              np.linspace(0.01, 0.99, 50), \
              np.sort(np.append(np.geomspace(1e-5,1e0,num=6),np.geomspace(5e-5,5e-1,num=5))), \
              [False,True], \
              [False,True], \
              np.linspace(0.1, 2, 20), \
              np.sort(np.append(np.geomspace(1e-6,1e1,num=8),np.geomspace(5e-6,5e0,num=7)))]
    
    #possible learning rates, used for params==learning_rate_init and tol
    learning_rates = ['constant','invscaling','adaptive']
    
    #detault classifier arguments
    base_args = {'solver':'sgd', 'early_stopping':True}
    
    #cycle through params and pass to mlp_explore_param for the hard work
    output = []
    for i, param in enumerate(params):
        
        #for these params, cycle value through the possible learning rates
        if param=='learning_rate_init' or param=='tol':
            for rate in learning_rates:
                args = {**base_args, **{'learning_rate':rate}}
                df_totals = mlp_explore_param(param, values[i], X_train, y_train, X_test, y_test, args=args, redo=redo)
                output.append(df_totals)
        
        #this param is only used for learning_rate=invscaling, so ensure that
        elif param=='power_t':
            args = {**base_args, **{'learning_rate_init':0.1, 'learning_rate':'invscaling'}}
            df_totals = mlp_explore_param(param, values[i], X_train, y_train, X_test, y_test, redo=redo)
            output.append(df_totals)
            
        else:
            df_totals = mlp_explore_param(param, values[i], X_train, y_train, X_test, y_test, redo=redo)
            output.append(df_totals)
        
    return output, param_titles

def mlp_hyper_awesome(tables, param_titles, do_display=True):
    '''
    Name:
        mlp_hyper_awesome
    
    Purpose: 
        Take tables from mlp_explore_params, choose "best" parameter value 
        based on accuracy and f1 score, output results in a dataframe, 
        and format/display all the tables
    
    Parameters: 
        2 Required Inputs:
    
        tables = list containing Pandas DataFrames from mlp_explore_params
        param_titles = string array containing the associated parameter names
        
        1 Optional Settings:
        
        do_display = Boolean, default=True. Set to display all tables.
        
    
    Returns: 
        Out: Pandas DataFrame with "best" value for each pameter based on 
        highest average of accuracy and f1 score. 
    ''' 
    
    best_values = []
    best_acc = np.zeros(np.size(param_titles))
    best_f1 = np.zeros(np.size(param_titles))
    
    for i, table in enumerate(tables):
        acc = table['Overall Accuracy'].values
        f1 = table['Average F-Meas'].values
        score = acc
        runtime = table['Run Time'].values
        values = table.iloc[:,0].values

        highscore = np.argwhere(score == np.amax(score)).flatten()
        lowtime = np.argwhere(runtime[highscore] == np.amin(runtime[highscore])).flatten()
        ind = highscore[lowtime]
        value = values[ind]
        
        best_values.append(str(value))
        best_acc[i] = acc[highscore[lowtime]]
        best_f1[i] = f1[highscore[lowtime]]
        
        if do_display==True:
            print('')
            print('"Best" value = '+str(value)+' for param = '+param_titles[i])
            display(table.style.apply(c.color_max, axis=0).apply(c.color_min).apply(\
            lambda x: ['background: yellow' if x.name == ind else '' for i in x], axis=1))
    
    #best_values = np.array(best_values).flatten()
    output = pd.DataFrame(index=param_titles, data={'Best Value':best_values, 'F1':best_f1, 'Overall Accuracy':best_acc})
    
    if do_display==True:
        print('Best values from the tables above...')
        display(output)
    
    return output
