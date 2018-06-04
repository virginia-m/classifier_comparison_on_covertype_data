from .evaluations import construct_confusion_matrix, \
                        plot_confusion_matrix, \
                        evaluate_classifier_predictions, \
                        cross_validate_classifier, \
                        format_classifier_performance, \
                        split_dataset
                        
from .mcmetrics import metrics_wrapper, \
					   confusion_matrix, \
					   color_good_bad, \
					   highlight_max, \
					   color_max, \
					   color_min

from .preprocessing import get_features_and_labels, \
                           load_covertype_data, compress_and_engineer_features
                       
from .mlp_explorer import read_covtype, \
						  mlp_explore_param, \
						  mlp_explore_params, \
						  mlp_hyper_awesome

