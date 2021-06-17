from collections import defaultdict


def get_indipendence(df, prediction_col, cols):
    """
    Args:
        df (DataFrame):Input data
        prediction_col (str): name of the column containing the prediction
        positive_value (str): positive outcome for the binary classification task
        cols (list): List of features that need to be inspected

    Returns:
         dictionaries whose the key are the columns and whose values are
         the calculated probabilities within each class. One dictionary
         for the positive class, and one for the negative one

         P( Y = 1 | A = a ) = P( Y = 1 | A = b )
            * at most with a small tolerance

            * P ( Y = 1 | A = a) = ( P ( Y = 1) \intersection P( A = a) ) / P( A = a )
    """
    dic_indipendence = defaultdict(list)

    # get the values from the target
    target_classes = df[prediction_col].unique()

    for col in cols:

        # the outcome is fixed and it is based on the predictor

        # get the classes within the current column
        classes = df[col].unique()

        for c in classes:

            # numerator = P ( Y = 1 ) \intersection P ( A = a)
            # for both numerator and denominator I am not considering the division by len(df)
            # denominator = P ( A = a)

            for target in target_classes:
                numerator = len(df[(df[prediction_col] == target) & (df[col] == c)])
                denom = len(df[ df[col] == c])

                # set two different list for each col, one for each class (0/1)
                key = col + "-" + str(target)

                dic_indipendence[key].append(numerator / denom)

    return dic_indipendence


def get_separation(df, target_col, prediction_col, positive_value, cols):
    """
    Args:
        df (DataFrame):Input data
        target_col (str): ground truth of what we want to predict
        prediction_col (str): name of the column containing the prediction
        positive_value (str): positive outcome for the binary classification task
        cols (list): List of features that need to be inspected

    Returns:
         dictionaries whose the key are the columns and whose values are
         the calculated probabilities within each class. One dictionary
         for the positive class, and one for the negative one

         P( R = 1 | Y = 1, A = a ) = P( R = 1 | Y = 1, A = b )
            * at most with a small tolerance
            * R = target
            * Y = predictor

         P( R = 1 | Y = 1, A = a ) = ( P ( R = 1 ) \intersection P ( Y = 1 ) * P ( A = a) ) / P ( Y = 1 ) * P ( A = a)
    """
    dic_separation = defaultdict(list)

    # get the values from the target
    target_classes = df[target_col].unique()

    print()

    for col in cols:

        # the outcome is fixed and it is based on the predictor

        # get the classes within the current column
        classes = df[col].unique()

        for c in classes:
            # numerator = P ( R = 1 ) \intersection P ( Y = 1 ) * P ( A = a) )
            # denom = P ( Y = 1 ) * P ( A = a)

            # for both numerator and denominator I am not considering the division by len(df)

            for target in target_classes:
                numerator = len(df[(df[target_col] == positive_value) & (df[prediction_col] == target) & (df[col] == c)])
                denom = len(df[(df[prediction_col] == positive_value) & (df[col] == c)])

                # set two different list for each col, one for each class (0/1)
                key = col + "-" + str(target)
                dic_separation[key].append(numerator / denom)

    return dic_separation

