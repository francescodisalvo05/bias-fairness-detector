
import pandas as pd


def compact_binary_features(dir_dataset, features, new_feature, new_path, neutral=None):
    """
    Args:
         dir_dataset (str) : path of the dataset
         features (list) : list of binary features to merge
         new_feature (str) : name of the new categorical feature
         new_path (str) : directory + filename (e.g. 'data/cleaned_dataset.csv')
         neutral (str) : str for a neutral combination (if all "non" positive have a meaning)
    Returns:
         Save the compacted dataset, for which we have a new column called
         "race", in which we add the race, given by the binary features
    """
    df = pd.read_csv(dir_dataset)
    new_feature_list = []

    # iterrows is very expansive
    # to do : try to use some more efficient workaround
    for index, row in df.iterrows():
        found = False
        for feature in features:
            if row[feature] == 1:
                new_feature_list.append(feature)
                found = True

        if not found:
            new_feature_list.append(neutral)

    for feature in features:
        df.drop(columns=[feature], inplace=True)

    df[new_feature] = pd.Series(new_feature_list)
    df.to_csv(new_path, index=False)