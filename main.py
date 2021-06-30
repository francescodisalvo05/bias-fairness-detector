from utils.bias_measures import *
from utils.fairness_measures import *
from utils.model import *

from argparse import ArgumentParser


def bias(df_main, sensitive_list):
    print("Gini :  ", get_gini_index(df_main,  sensitive_list))
    print("Shannon :  ", get_shannon_index(df_main,  sensitive_list))
    print("Simpson :  ", get_simpson_index(df_main,  sensitive_list))
    print("Imbalanced :  ", get_imbalance_ratio(df_main,  sensitive_list))


def fairness(df_main, target_col, prediction_col, positive_outcome, sensitive_list):

    print("Indipendence :  ", get_indipendence(df_main, prediction_col, sensitive_list))
    print("Separation :  ", get_separation(df_main, target_col, prediction_col, positive_outcome, sensitive_list))
    print("Sufficiency :  ", get_sufficiency(df_main, target_col, prediction_col, positive_outcome, sensitive_list))


def main():
    parser = ArgumentParser()

    parser.add_argument('-b', '--bias', type=bool, default=True, help='Print bias measures')
    parser.add_argument('-f', '--fairness', type=bool, default=True, help='Print fairness measures')
    parser.add_argument('-d', '--dataset', type=str, default='data/propublica_data_for_fairml_cleaned.csv', help='Directory of the dataset')
    parser.add_argument('-s', '--sensitive_attr', nargs='+', help='List of sensitive attributes (e.g. --sensitive_attr race gender', required=True)

    parsed_args = parser.parse_args()

    #df = pd.read_csv(parsed_args.dataset)

    #print(df)

    # df_with_prediction = get_prediction(df.drop(columns=['race']), 'score_factor')
    # fairness(df_with_prediction, 'score_factor', 'prediction', 1, ['Asian'])

if __name__ == '__main__':
    main()

