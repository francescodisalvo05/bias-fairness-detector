from utils.bias_measures import *
from utils.fairness_measures import *


def bias(df_main, sensitive_list):
    print("Gini :  ", get_gini_index(df_main,  sensitive_list))
    print("Shannon :  ", get_shannon_index(df_main,  sensitive_list))
    print("Simpson :  ", get_simpson_index(df_main,  sensitive_list))
    print("Imbalanced :  ", get_imbalance_ratio(df_main,  sensitive_list))


def fairness(df_main, prediction_col, positive_outcome, sensitive_list):
    # to do : ML model for the prediction over Two_yr_Recidivism
    # it should be the "predictor"
    print("Indipendence :  ", get_indipendence(df_main, prediction_col, sensitive_list))

    # >> it will be both one because predictor and target are the same
    print("Separation :  ", get_separation(df_main, prediction_col, prediction_col, positive_outcome, sensitive_list))
    # >> it will be both one because predictor and target are the same
    print("Sufficiency :  ", get_sufficiency(df_main, prediction_col, prediction_col, positive_outcome, sensitive_list))


if __name__ == '__main__':

    df = pd.read_csv('data/propublica_data_for_fairml.csv')

    # to do : update the DF in order to have just one "race" column instead of
    # sensitive_cols = ['Female', 'Age_Above_FourtyFive', 'Age_Below_TwentyFive']
    fairness(df, 'Two_yr_Recidivism', 1, ['Female'])
