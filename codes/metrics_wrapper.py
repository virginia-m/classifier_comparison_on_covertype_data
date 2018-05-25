import numpy as np
import pandas as pd
from IPython.core.display import display

def confusion_matrix(actual, predicted, classes):
    '''
    Name:
        confusion_matrix
    
    Purpose: 
        Generate confusion matrix from actual and predicted class labels
    
    Parameters: 
        3 Required Inputs:
        1 (actual) = NumPy array with actual class labels
        2 (predicted) = NumPy array with predicted class lables 
        3 (classes) = NumPy array with unique class names
    
    Returns: 
        1 Output variable:        
        1 (confusion) = NumPy array with confusion matrix, where the actual 
            labels are on the rows and the predicted labels are on the columns
    ''' 

    nclass = np.shape(classes)[0]
    confusion = np.zeros([nclass,nclass], dtype=int)

    for i, this_class in enumerate(classes):
        these_predicts = predicted[np.where(actual==this_class)]
        for j, that_class in enumerate(classes):
            confusion[i,j] = np.count_nonzero(these_predicts==that_class)

    return confusion

def classy_metrics(conf):
    '''
    Name:
        classy_metrics
        
    Purpose: 
        Obtain accuracy, precision, recall, and f-measure from a confusion matrix
        
    Parameters: 
        1 input parameter:
        1 (conf) = NumPy array containing confusion matrix, produced by confusion_matrix()
    
    Returns: 
        7 output variables:
        1 (acc) = overall accuracy (total correct) / (total)
        2 (pre) = average precision across classes
        3 (rec) = average recall across classes
        4 (fme) = average f-measure across classes
        5 (class_pre) = array of class-specific precisions
        6 (class_rec) = array of class-specific recalls
        7 (class_fme) = array of class-specific f-measures
    ''' 

    #calculate true positions, true negatives, false positives, and false negatives
    true_pos = np.diagonal(conf) #diagonal of confusion matrix 
    false_pos = np.sum(conf, axis=0) - true_pos #column totals - diaganol
    false_neg = np.sum(conf, axis=1) - true_pos #row totals - diaganol
    true_neg = np.sum(conf)-(true_pos+false_pos+false_neg) #total - (TP+FP+FN)

    #class accuracies, precisions, and recalls. 
    old_settings = np.seterr(divide='ignore',invalid='ignore') #in case any divide by 0 instances
    class_acc = np.divide((true_pos + true_neg), (true_pos + true_neg + false_pos + false_neg))
    class_pre = np.divide(true_pos, (true_pos+false_pos))
    class_rec = np.divide(true_pos, (true_pos+false_neg))
    class_fme = np.divide((2*class_rec*class_pre), (class_rec+class_pre))
    np.seterr(**old_settings)
    
    np.nan_to_num(class_acc ,copy=False) #in case there were any divide by 0s
    np.nan_to_num(class_pre ,copy=False)
    np.nan_to_num(class_rec ,copy=False)
    np.nan_to_num(class_fme, copy=False)

    acc = np.sum(np.diagonal(conf)) / np.sum(conf)

    pre = np.mean(class_pre)
    rec = np.mean(class_rec)
    fme = np.mean(class_fme)

    return acc, pre, rec, fme, class_pre, class_rec, class_fme

def format_metrics(all_acc,all_pre,all_rec,all_fm,all_cl_pre,all_cl_rec,all_cl_fm,all_conf,classes):

    total_metrics = pd.DataFrame({'Overall Accuracy':np.round(np.mean(all_acc), decimals=2), \
                           'Average Precision':np.round(np.mean(all_pre), decimals=2), \
                           'Average Recall':np.round(np.mean(all_rec), decimals=2), \
                           'Average F-Meas':np.round(np.mean(all_fm), decimals=2)}, \
                            index=['Class-Averaged or Overall:'])              
            
    class_metrics = pd.DataFrame({'Class':classes, \
                               'Precision':np.round(all_cl_pre, decimals=2), \
                               'Recall':np.round(all_cl_rec, decimals=2), \
                               'F-Meas':np.round(all_cl_fm, decimals=2)})
    class_metrics.rename_axis('Confusion Matrix Index', inplace=True)

    ind = np.arange(np.shape(classes)[0])
    confusion_frame = pd.DataFrame(all_conf, index=ind, columns=ind)
    confusion_frame.rename_axis('Actual', inplace=True)
    confusion_frame.rename_axis('Predicted', inplace=True, axis='columns')
    
    return total_metrics, class_metrics, confusion_frame

'''
These two functions are used to highlight particular cells of the 
Pandas DataFrames that will be displayed to show my results
'''
def color_good_bad(val):
    color = 'black'
    if val < 0.3: color = 'red'
    if val > 0.7: color = 'blue'
    return 'color: %s' % color

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]
    
def color_max(s):
    is_max = s == s.max()
    return ['color: red' if v else '' for v in is_max]    

def metrics_wrapper(conf, classes, do_display=False):

    aconf=np.mean(conf, axis=0) if conf.ndim==3 else conf

    acc, pre, rec, fme, class_pre, class_rec, class_fme = classy_metrics(aconf)
    total_metrics, class_metrics, confusion_frame = format_metrics(acc, pre, rec, fme, class_pre, class_rec, class_fme, aconf, classes)
    
    if do_display==True:
        print('Average/overall metrics:')
        display(total_metrics) 
        print('Class-specific metrics:')
        display(class_metrics.style.applymap(color_good_bad, subset=['Precision', 'Recall', 'F-Meas']))
        print('Confusion matrix (yellow = col max; red = row max):')
        display(confusion_frame.style.apply(highlight_max, axis=0).apply(color_max, axis=1))
    
    return total_metrics, class_metrics, confusion_frame
