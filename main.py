from utils.bias_measures import *

import pandas as pd

from prettytable import PrettyTable

from argparse import ArgumentParser


def printTable(title, measures, header, sensitive_list):

    # construct the header
    # -- first element
    table = PrettyTable(header)
    # add the title to the table
    table.title = title

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
    printTable('Bias measures',measures,header, sensitive_list)


def main():

    parser = ArgumentParser()

    # work in progress
    # parser.add_argument('-b', '--bias', type=bool, default=True, help='Print bias measures')
    # parser.add_argument('-f', '--fairness', type=bool, default=True, help='Print fairness measures')
    # parser.add_argument('-t', '--target', help='Target of the prediction', default='score_factor')
    # parser.add_argument('-p', '--positive_outcome', help='Positive outcome', default=1)

    parser.add_argument('-d', '--dataset', type=str, default='data/propublica_data_cleaned.csv',
                        help='Directory of the dataset')

    parser.add_argument('-s', '--sensitive_attr', nargs='+', help='List of sensitive attributes (e.g. --sensitive_attr race gender',
                              default=['Ethnicity','Age','Female'])

    parsed_args = parser.parse_args()
    df = pd.read_csv(parsed_args.dataset)
    bias(df, parsed_args.sensitive_attr)


if __name__ == '__main__':
    main()



