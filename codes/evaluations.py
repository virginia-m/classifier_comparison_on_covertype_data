import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

def construct_confusion_matrix(actual, predicted, dim=7):
    '''
        This function constructs the confusion matrix for a given input
        of actual and predicted class vectors with integers as class labels.
        
        Parameters
        ----------------------
        actual : array, dtype int
            A 1D row vector of length N containing the actual class labels
        
        predicted : array, dtype int
            A 1D row vector of length N containing the predicted class labels
        
        dim : integer
            The (one-axis) size of the confusion matrix. This will depend on how
            many classes there are in the data.
            
        Returns
        ----------------------
        confusion_matrix : array
            The confusion matrix constructed from actual and predicted class
            labels with dimension (N, N).
    '''
    confusion_matrix = np.zeros((dim, dim))
    for row, col in zip(actual, predicted):
        confusion_matrix[row, col] += 1
    return confusion_matrix

def plot_confusion_matrix(confusion_matrices):
    '''
        This function plots the 2D confusion matrix for a given input confusion
        matrix. This can be a single or multiple matrices.
        
        Parameters
        ----------------------
        confusion_matrices : list
            Python list containing a set of confusion matrices.
            
        Returns
        ----------------------
        fig : matplotlib figure
            The matplotlib figure instance on which the confusion matrix has been
            plotted.
    '''
    num_plots = len(confusion_matrices)
    
    fig = plt.figure(figsize=(15, 7.5))
    gs = GridSpec(1, num_plots)
    
    for i in range(num_plots):
        ax = plt.subplot(gs[0, i])
        confusion_matrix = confusion_matrices[i]
        ax.imshow(confusion_matrix, vmin=0)
        
    return fig

def evaluate_classifier_predictions(confusion_matrix):
    """
        This function takes the confusion matrix generated from a classifier prediction
        and calculates relevant performance metrics: accuracy, precision, recall and the
        f1 score.
        
        Parameters
        ----------------------
        confusion_matrix : array
            The confusion matrix for a classifier prediction.
            
        Returns
        ----------------------
        accuracy : number
            The accuracy of the predictions.
        
        precision, recall, f1_score : array
            1D row vectors containing the precision, recall and F1 scores of the predictions.
            
        Note
        ----------------------
        When calculating precision and recall, we subsitute zero-count values with small
        numbers to avoid division by zeros.
        
    """
    c = np.copy(confusion_matrix)
    diag = np.diagonal(c)

    # Accuracy: number of correct predictions over total number of points (overall metric)
    accuracy = np.sum(diag) / np.sum(c)
    
    # Precision: number of correct predictions over total number of predictions for this class
    c_sum = np.sum(c, axis=0)
    c_sum[c_sum == 0] = 1e-20
    precision = diag / c_sum
    
    # Recall: number of correct predictions over number of occurrences of this prediction
    r_sum = np.sum(c, axis=1)
    r_sum[r_sum == 0] = 1e-20
    recall = diag / r_sum
    
    # F1 score: harmonic mean (centre of mass) between recall and precision
    f1_score = precision * recall / (precision + recall) * 2
    
    return accuracy, precision, recall, f1_score


def format_classifier_performance(confusion_matrix):
    """
        This function formats the mean classifier performance by calling the
        *evaluate_classifier_predictions* function with a given confusion matrix.
    """
    results = """
    Overall Accuracy: {:.3f}
    Mean Precision:   {:.3f} +/- {:.3f}
    Mean Recall:      {:.3f} +/- {:.3f}
    Mean F1 Score:    {:.3f} +/- {:.3f}
    """
    
    a, p, r, f = evaluate_classifier_predictions(confusion_matrix)
    return results.format(a, np.mean(p), np.std(p), np.mean(r), np.std(r), np.mean(f), np.std(f))


def generate_cross_validation_datasets(full_data, labels, kfold=10, training_percentage=0.9):
    """
        This function generates a number of splitted datasets for training and validation
        using an input dataset.
        
        Parameters
        ----------------------
        full_data : sparse array
            The dataset from which to draw samples for training and validation.
            
        kfold : integer
            The number of k-folds in the cross-validation. This determines how many
            datasets are produced.
        
        training_percentage : number
            The splitting ratio between training and validation data.
            
        Returns
        ----------------------
        datasets : list
            A Python list containing two-element lists with training and validation
            datasets, stored in numpy arrays.
        
        
    """
    num_rows = full_data.shape[0]
    row_indices = np.arange(num_rows)
    split_row = int(num_rows * training_percentage)
    
    # Create an empty list to hold pairs of training and validation data
    datasets = []
    for i in range(kfold):
        # Generate a random shuffle of the row indices.
        # Note: this operation shuffles in place.
        np.random.shuffle(row_indices)
        # Get the training and validation rows
        training_rows, validation_rows = np.split(row_indices, [split_row])
        
        # Now extract the training and validation rows from the input datset:
        training_data, validation_data = [], []
        for dataset, rows in zip([training_data, validation_data], [training_rows, validation_rows]):
            for row in rows:
                dataset.append(full_data[row, :])
            
        datasets.append(
            [labels[training_rows], np.vstack(training_data),
             labels[validation_rows], np.vstack(validation_data)])
    
    return datasets

def class_count_split(X, y, count=250):
    '''
    Parameters:
        X = data
        y = labels
        
        count: number of each class to keep in training set, default 250, only 
                used if STYLE='equal'
                    
    Returns:
        X_train, y_train, X_test, y_test
    '''   
    
    #unique classes
    classes = np.unique(y)
    
    #empty array to store random pick indices 
    picks = np.empty([count, np.shape(classes)[0]])

    #loop through each class and randomly select <count> representatives
    for i, this_class in enumerate(classes):
        want = np.where(y == this_class)[0]
        picks[:,i] = np.random.choice(want,size=count, replace=False)

    #sort pick indices by the order they appear in the original table
    picks = np.sort(picks.flatten()).astype(int)
    
    X_test = X[picks,:]
    y_test = y[picks]
    
    X_train = np.delete(X,picks,axis=0)
    y_train = np.delete(y,picks)  
        
    return X_train, y_train, X_test, y_test
    
def split_dataset(X, y, style='prop', count=250, test_size=0.1, regularize=False):
    '''
    Parameters:
        X = data
        y = labels
        
        style: 'prop' = classes proportionally represented in test dataset based
                        on proportions in training dataseg
                'equal' = equal number of class counts in training set
                'random' = random test/training partition
        
        count: number of each class to keep in training set, default 250, only 
                used if STYLE='equal'
        
        test_size: fraction of dataset used for test set, default 0.1, only used 
                    if STYLE='prop' or 'random'
                    
    Returns:
        X_train, y_train, X_test, y_test
    '''
    
    if style=='equal':
        X_train, y_train, X_test, y_test = class_count_split(X,y,count=count)
        
    if style=='prop':
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
    if style=='random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
    if regularize==True:
        scaler = StandardScaler()  
        scaler.fit(X_train) # supposed to fit only on training data 
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test) # apply same transformation to test data
            
    return X_train, y_train, X_test, y_test    

def cross_validate_classifier(classifier, full_data, labels, kfold=10, training_percentage=0.9, kwargs=None,
                              count=250, test_size=0.1, style='prop', regularize=False):
    """
        This function evaluates a classifier using k-fold cross-validation
        on a given dataset and by calculating confusion matrices for each fold.
        
        This function generates a number of splitted datasets for training and validation
        using an input dataset.
        
        Parameters
        ----------------------
        classifier : Classifier instance
            An instance of a classifier class that contains two methods named *fit* and
            *predict*, abiding to scikit-learn naming conventions.
        
        full_data : numpy array
            The dataset from which to draw samples for training and validation.
        
        kfold : integer, optional
            The number of k-folds in the cross-validation.
        
        training_percentage : number, optional
            The splitting ratio between training and validation data.
            
        kwargs : dictionary, optional
            A dictionary with keyword arguments to be passed to the classifier's
            constructor. Defaults to None.
            
            
        Returns
        ----------------------
        confusion_matrices : list
            Python list containing the confusion matrices for each fold in the cross-validation.
            
        runtimes : NumPy array
        	Array of runtimes for each cross-validation cycle, including the time for 
        	splitting the data into training and test sets
    """
    if kwargs is None:
        kwargs = {}

    confusion_matrices = []
    classifiers = []
    runtimes = np.zeros(kfold)
    
    for i in range(kfold):
        start_time = time.time()
    	
    	#split dataset into training and test set
        train_data, train_labels, validation_data, validation_labels = \
        split_dataset(full_data, labels, style=style, count=count, test_size=test_size,
                      regularize=regularize)
                
        # Fit the classifier using the training data, then predict using the validation data.
        _classifier = classifier(**kwargs)
        _classifier = _classifier.fit(train_data, train_labels)
        predictions = _classifier.predict(validation_data)
        
        # Compute the confusion matrix and store the results
        confusion_matrix = construct_confusion_matrix(validation_labels, predictions)
        confusion_matrices.append(confusion_matrix)
        classifiers.append(_classifier)
        runtimes.append(time.time() - start_time)
    
    return confusion_matrices, runtimes, classifiers

