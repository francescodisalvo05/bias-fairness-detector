from utils.bias_measures import *
from utils.fairness_measures import *
from utils.model import *

from prettytable import PrettyTable

from argparse import ArgumentParser

from utils.dataset import compact_binary_features


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
    

def fairness(df_main, target_col, prediction_col, positive_outcome, sensitive_list):

    print("Indipendence :  ", get_indipendence(df_main, prediction_col, sensitive_list))
    print("Separation :  ", get_separation(df_main, target_col, prediction_col, positive_outcome, sensitive_list))
    print("Sufficiency :  ", get_sufficiency(df_main, target_col, prediction_col, positive_outcome, sensitive_list))


def main():
    sensitive_list = ['Age_Above_FourtyFive', 'Age_Below_TwentyFive', 'African_American',
                      'Asian', 'Hispanic', 'Native_American', 'Other', 'Female']

    parser = ArgumentParser()

    parser.add_argument('-b', '--bias', type=bool, default=True, help='Print bias measures')
    parser.add_argument('-f', '--fairness', type=bool, default=True, help='Print fairness measures')

    parser.add_argument('-d', '--dataset', type=str, default='data/propublica_data_cleaned.csv',
                        help='Directory of the dataset')

    # >> remove default and set required
    parser.add_argument('-s', '--sensitive_attr', nargs='+', help='List of sensitive attributes (e.g. --sensitive_attr race gender',
                              default=['Ethnicity','Age','Female'])
    parser.add_argument('-t', '--target', help='Target of the prediction', default='score_factor')
    parser.add_argument('-p', '--positive_outcome', help='Positive outcome', default=1)

    parsed_args = parser.parse_args()


    # list_s = ['Age_Above_FourtyFive','Age_Below_TwentyFive']
    # compact_binary_features(parsed_args.dataset, list_s, 'Age', 'data/propublica_data_cleaned.csv', '25-45')

    df = pd.read_csv(parsed_args.dataset)
    bias(df, parsed_args.sensitive_attr)

    #df_with_prediction = get_prediction(df.drop(columns=parsed_args.sensitive_attr), parsed_args.target)
    #fairness(df_with_prediction, parsed_args.target, 'prediction', 1, parsed_args.sensitive_attr)


if __name__ == '__main__':
    main()



