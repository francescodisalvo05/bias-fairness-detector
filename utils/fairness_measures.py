from collections import defaultdict


def get_indipendence(df, predictor, positive_value, cols):
    """
    Args:
        df (DataFrame):Input data
        cols (list): List of features that need to be inspected

    Returns:
         dictionary where the key are the columns and the values are
         the 
         
         P( Y = 1 | A = a ) = P( Y = 1 | A = b )
            * at most with a small tolerance

            * P ( Y = 1 | A = a) = ( P ( Y = 1) \intersection P( A = a) ) / P( A = a )
    """
    dic_indipendence = defaultdict(list)

    for col in cols:

        # the outcome is fixed and it is based on the predictor

        # get the classes within the current column
        classes = df[col].unique()

        for c in classes:

            # numerator = P ( Y = 1 | A = a)
            numerator = len(df[(df[predictor] == positive_value) & (df[predictor] == positive_value)]) / len(df)
            denom = len(df[ df[col] == c] )

            dic_indipendence[col].append(numerator / denom)

    return dic_indipendence


