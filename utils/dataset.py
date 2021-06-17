
def clean_compass_data(df):
    """
    Args:
         Compass dataset
    Returns:
         Save the cleaned dataset, for which we have a new column called
         "race", in which we add the race, given by the binary features
    """

    binary_race_features = ['African_American','Asian','Hispanic','Native_American','Other']

    def getRace(African_American,Asian,Hispanic,Native_American,Other):

        if African_American == 1:
            return 'African_American'
        elif Asian == 1:
            return 'Asian'
        elif Hispanic == 1:
            return 'Hispanic'
        elif Native_American == 1:
            return 'Native_American'
        elif Other == 1:
            return 'Other'

    df['race'] = df.apply(lambda row: getRace(row['African_American'],row['Asian'],row['Hispanic'],row['Native_American'],row['Other']), axis=1)

    df.to_csv('data/propublica_data_for_fairml_cleaned.csv', index=False)