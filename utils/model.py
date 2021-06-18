from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_prediction(df, target):
    """
    Args:
        df (DataFrame) : dataframe from which we need
                         make the prediction
        target (str) : name of the feature that we want
                       to predict
    """

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y)

    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    X_test['prediction'] = y_pred

    return X_test