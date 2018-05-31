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

    cover_types = [
        'Spruce/fir',
        'Lodgepole pine',
        'Ponderosa pine',
        'Cottonwood/willow',
        'Aspen',
        'Douglas-fir',
        'Krummholz'
    ]

    return pd.read_csv(
        join('data','covtype.data'), header=None, names=column_names, dtype=column_types)

def create_features_and_labels(data):
    """
        This function takes the forest covertype data in form of a pandas dataframe
        performs some feature conversion and engineering, and returns a numpy array with
        the new features and labels.
    """
    # First, separate features and labels
    features = data.loc[:, :'Cover_type'].values
    labels = data['Cover_type'].values - 1
    
    # Next, compress the binary columns (wilderness area and soil)
    compressed_warea = _compress_binary_columns(
        data[['Wilderness_area_{}'.format(i) for i in range(4)]].values
    )
    compressed_soil_types = _compress_binary_columns(
        data[['Soil_type_{}'.format(i) for i in range(40)]].values
    )
    # Modify the features array
    features = np.hstack((
        features[:, :10],                # Continuous variables
        compressed_warea[:, np.newaxis], # The new wilderness areas
        compressed_soil_types[:, np.newaxis],  # The new soil types
    ))
    
    # Now engineer new features
    engineered_features = _engineer_features(data, compressed_warea, compressed_soil_types)
    
    # Put everything together
    full_features = np.hstack((features, engineered_features))
    
    return full_features, labels

def _engineer_features(data, warea, soil_types):
    """
        This function engineers a set of new features based on combinations
        of features in the incoming dataframe. Combinations are chosen
        according to how they can improve class separation.
    """
    new_features = [
        # Elevation + distance to nearest road
        data['Elevation'].values  + data['Horiz_dist_to_nearest_road'].values,
        # Elevation - distance to nearest road
        data['Elevation'].values  - data['Horiz_dist_to_nearest_road'].values,
        # Elevation + distance to nearest wildfire ignition point
        data['Elevation'].values + data['Horiz_dist_to_fire_ignition_point'].values,
        # Elevation - distance to nearest wildfire ignition point
        data['Elevation'].values - data['Horiz_dist_to_fire_ignition_point'].values,
        # Product of elevation and soil types
        np.sqrt(data['Elevation'] * (soil_types + 1)),
        # Product of elevation and wilderness area
        data['Elevation'] * (warea + 1),
        # Product of wilderness area and soil types
        np.log((warea+1)/4) * (soil_types + 1)  
    ]
     
    return np.hstack([f[:, np.newaxis] for f in new_features])

def _compress_binary_columns(columns):
    """
        This function compresses a set of mutually exclusive binary columns
        to a single column that holds the column indices where the binary
        columns were non-zero.
        
        Note: for arrays with only two values (0 and 1), this is simply an
        argmax along the column axis.
    """
    return np.argmax(columns, axis=1)

