from utils.bias_measures import *
from utils.fairness_measures import *
from utils.model import *

from prettytable import PrettyTable

from argparse import ArgumentParser


def printTable(measures, header, sensitive_list):

    # construct the header
    # -- first element
    table = PrettyTable(header)

    for i,feature in enumerate(sensitive_list):

        # construct the table
        # -- the first element column will be the feature
        # -- the remaining ones will be the measures
        row = [feature]
        for measure in measures:
            row.append(measure[i])

        table.add_row(row)

    print(table)


def bias(df_main, sensitive_list):
    gini = get_gini_index(df_main,  sensitive_list)
    shannon = get_shannon_index(df_main,  sensitive_list)
    simpson = get_simpson_index(df_main,  sensitive_list)
    ir = get_imbalance_ratio(df_main,  sensitive_list)

    measures = [gini, shannon, simpson, ir]
    header =  ['Feature','Gini', 'Shannon', 'Simpson', 'Imbalanced Ratio']
    printTable(measures,header, sensitive_list)




def fairness(df_main, target_col, prediction_col, positive_outcome, sensitive_list):

    print("Indipendence :  ", get_indipendence(df_main, prediction_col, sensitive_list))
    print("Separation :  ", get_separation(df_main, target_col, prediction_col, positive_outcome, sensitive_list))
    print("Sufficiency :  ", get_sufficiency(df_main, target_col, prediction_col, positive_outcome, sensitive_list))


def main():

    parser = ArgumentParser()

    parser.add_argument('-b', '--bias', type=bool, default=True, help='Print bias measures')
    parser.add_argument('-f', '--fairness', type=bool, default=True, help='Print fairness measures')
    parser.add_argument('-d', '--dataset', type=str, default='data/propublica_data_for_fairml_cleaned.csv', help='Directory of the dataset')

    # >> remove default and set required
    parser.add_argument('-s', '--sensitive_attr', nargs='+', help='List of sensitive attributes (e.g. --sensitive_attr race gender',
                              default=['Asian','African_American'])
    parser.add_argument('-t', '--target', help='Target of the prediction', default='score_factor')
    parser.add_argument('-p', '--positive_outcome', help='Positive outcome', default=1)

    parsed_args = parser.parse_args()

    df = pd.read_csv(parsed_args.dataset)
    #df_with_prediction = get_prediction(df.drop(columns=parsed_args.sensitive_attr), parsed_args.target)
    #fairness(df_with_prediction, parsed_args.target, 'prediction', 1, parsed_args.sensitive_attr)

    bias(df, parsed_args.sensitive_attr)

if __name__ == '__main__':
    main()

