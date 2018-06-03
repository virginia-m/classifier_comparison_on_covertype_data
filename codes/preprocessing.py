from __future__ import print_function, division
from os.path import join
import numpy as np
import pandas as pd


def load_covertype_data():
    """
        This function loads the covertype data and stores it in a pandas
        dataframe object, with manually specified column names.
    """
    column_names = (
        [
        'Elevation',
        'Aspect_angle',
        'Slope',
        'Horiz_dist_to_nearest_water',
        'Vert_dist_to_nearest_water',
        'Horiz_dist_to_nearest_road',
        'Incident_sunlight_at_9am',
        'Incident_sunlight_at_12pm',
        'Incident_sunlight_at_3pm',
        'Horiz_dist_to_fire_ignition_point'    
        ]
        + ['Wilderness_area_{}'.format(i) for i in range(4)]
        + ['Soil_type_{}'.format(i) for i in range(40)]
        + ['Cover_type']
    )

    column_types = {
        name: np.float32 for name in column_names[:10]}.update(
        {name: np.int8 for name in column_names[10:]})

    return pd.read_csv(
        join('data','covtype.data'), header=None, names=column_names, dtype=column_types)

def get_features_and_labels(data):
    """
        This function takes the forest covertype data in form of a pandas dataframe
        performs some feature conversion and engineering, and returns a numpy array with
        the new features and labels.
    """
    # First, separate features and labels
    data_num = data.values
    return data_num[:, :-1], data_num[:, -1] - 1
    
def compress_and_engineer_features(data):
    """
        This function takes the original covertype dataset as input and compresses the
        binary columns, and engineers a handful of new features. The return is a dataframe
        containing all the features (old + new)
    """
    # Start by compressing the binary columns
    compressed_warea = _compress_binary_columns(
        data[['Wilderness_area_{}'.format(i) for i in range(4)]].values
    )
    compressed_soil_types = _compress_binary_columns(
        data[['Soil_type_{}'.format(i) for i in range(40)]].values
    )
    
    # Drop the corresponding columns in the dataframe, and add the new ones
    data_new = data.drop(list(data)[10:], axis=1)
    data_new.loc[:, 'Wilderness_area'] = compressed_warea
    data_new.loc[:, 'Soil_types'] = compressed_soil_types
    
    # Engineer new features and add them to the dataframe
    new_features = _engineer_features(data_new)
    for feature_name, feature_data in new_features:
        data_new.loc[:, feature_name] = pd.Series(feature_data, index=data_new.index)
        
    # Finally, add the cover type from the original data frame as last column:
    data_new.loc[:, 'Cover_type'] = pd.Series(data['Cover_type'].values, index=data_new.index)
        
    return data_new

def _engineer_features(data):
    """
        This function engineers a set of new features based on combinations
        of features in the incoming dataframe. Combinations are chosen
        according to how they can improve class separation.
    """
    new_features = [
        # Elevation + distance to nearest road
        ('Elevation_plus_road_dist',
         data['Elevation'].values  + data['Horiz_dist_to_nearest_road'].values),
        # Elevation - distance to nearest road
        ('Elevation_minus_road_dist',
         data['Elevation'].values  - data['Horiz_dist_to_nearest_road'].values),
        # Elevation + distance to nearest wildfire ignition point
        ('Elevation_plus_fire_dist',
         data['Elevation'].values + data['Horiz_dist_to_fire_ignition_point'].values),
        # Elevation - distance to nearest wildfire ignition point
        ('Elevation_minus_fire_dist',
         data['Elevation'].values - data['Horiz_dist_to_fire_ignition_point'].values),
        # Product of elevation and soil types
        ('Elevation_times_soil_types',
         np.sqrt(data['Elevation'] * (data['Soil_types'] + 1))),
        # Product of elevation and wilderness area
        ('Elevation_times_wArea',
         data['Elevation'] * (data['Wilderness_area'] + 1)),
        # Product of wilderness area and soil types
        ('wArea_times_soil_types',
        np.log((data['Wilderness_area']+1)/4) * (data['Soil_types'] + 1))
    ]
     
    return new_features

def _compress_binary_columns(columns):
    """
        This function compresses a set of mutually exclusive binary columns
        to a single column that holds the column indices where the binary
        columns were non-zero.
        
        Note: for arrays with only two values (0 and 1), this is simply an
        argmax along the column axis.
    """
    return np.argmax(columns, axis=1)

