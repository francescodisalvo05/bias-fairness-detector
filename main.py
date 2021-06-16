from utils.measures import *


def main(df_main, sensitive_list):
    print("Gini :  ", get_gini_index(df_main,  sensitive_list))
    print("Shannon :  ", get_shannon_index(df_main,  sensitive_list))
    print("Simpson :  ", get_simpson_index(df_main,  sensitive_list))
    print("Imbalanced :  ", get_imbalance_ratio(df_main,  sensitive_list))


if __name__ == '__main__':

    df = pd.read_csv('data/propublica_data_for_fairml.csv')

    # to do : update the DF in order to have just one "race" column instead of
    # sensitive_cols = ['Female', 'Age_Above_FourtyFive', 'Age_Below_TwentyFive']

    main(df, ['Female'])
