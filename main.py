from utils.bias_measures import *
from utils.fairness_measures import *
from utils.model import *


def bias(df_main, sensitive_list):
    print("Gini :  ", get_gini_index(df_main,  sensitive_list))
    print("Shannon :  ", get_shannon_index(df_main,  sensitive_list))
    print("Simpson :  ", get_simpson_index(df_main,  sensitive_list))
    print("Imbalanced :  ", get_imbalance_ratio(df_main,  sensitive_list))


def fairness(df_main, target_col, prediction_col, positive_outcome, sensitive_list):

    print("Indipendence :  ", get_indipendence(df_main, prediction_col, sensitive_list))
    print("Separation :  ", get_separation(df_main, target_col, prediction_col, positive_outcome, sensitive_list))
    print("Sufficiency :  ", get_sufficiency(df_main, target_col, prediction_col, positive_outcome, sensitive_list))


if __name__ == '__main__':

    df = pd.read_csv('data/propublica_data_for_fairml_cleaned.csv')
    df_with_prediction = get_prediction(df.drop(columns=['race']), 'score_factor')

    # sensitive_cols = ['Female', 'Age_Above_FourtyFive', 'Age_Below_TwentyFive','African_American','Asian','Hispanic','Native_American','Other']
    fairness(df_with_prediction, 'score_factor', 'prediction', 1, ['Asian'])
