import pandas as pd


def compact_non_binary_features(dir_dataset, features, new_feature, new_path, neutral=None):
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


if __name__ == '__main__':

    # call here the cleaning methods that you need
    # for example propublica's dataset had the following features:
    # -- Two_yr_Recidivism,Number_of_Priors,score_factor,Age_Above_FourtyFive,Age_Below_TwentyFive,
    # -- African_American,Asian,Hispanic,Native_American,Other,Female,Misdemeanor

    # in order to compact all these binary features I've used two times "compact_non_binary_features"

    # > list_s = ['African_American','Asian','Hispanic','Native_American','Other']
    # > compact_non_binary_features('../data/propublica_data_for_fairml.csv', list_s, 'Ethnicity', '../data/propublica_data_cleaned.csv', 'Caucasian')

    # > list_s = ['Age_Above_FourtyFive','Age_Below_TwentyFive']
    # > compact_non_binary_features('../data/propublica_data_cleaned.csv', list_s, 'Age', '../data/propublica_data_cleaned.csv', '25-45')

    # the final list of features will be:
    # -- Two_yr_Recidivism,Number_of_Priors,score_factor,Female,Misdemeanor,Ethnicity,Age

    pass