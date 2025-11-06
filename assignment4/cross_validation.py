''' This file contains the functions for performing cross-validation.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
import itertools
from typing import List, Tuple

import pandas as pd
import numpy as np

from data import load_data
from evaluate import accuracy
from model import init_perceptron, PERCEPTRON_VARIANTS


def cross_validation(
        cv_folds: List[pd.DataFrame], 
        perceptron_variant: str, 
        lr_values: list, 
        mu_values: list,
        epochs: int = 10) -> Tuple[dict, float]:
    '''
    Run cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of pandas DataFrames, corresponding to folds of the data. 
            The first column of each DataFrame, called "label", corresponds to y, 
            while the remaining columns are the features x.
        perceptron_variant (str): The variant of the perceptron algorithm to use
        lr_values (list): a list of learning rate hyperparameter values to try
        mu_values (list): a list of margin (mu) hyperparameter values to try
        epochs (int): how many epochs to train each model for. Defaults to 10

    Returns:
        dict: a dictionary with the best hyperparameters discovered during cross-validation
        float: the average cross-validation accuracy corresponding to the best hyperparameters

    Hints:
        - We've provided a helper function `init_perceptron()` in model.py to initialize your model. 
          You can call it with `model = init_perceptron(perceptron_variant, num_features=num_features, lr=lr, mu=mu)`
        - The python `itertools.product()` function returns the Cartesian product of multiple lists.
          You can call `itertools.product(lr_values, mu_values) to get all combinations as (lr, mu) tuples.
        - You can convert a pandas DataFrame to a numpy ndarray with `df.to_numpy()`
    '''

    best_hyperparams = {'lr': None, 'mu': None}
    best_avg_accuracy = 0


    num_features = cv_folds[0].shape[1] - 1  # -1 for the label column

    for lr, mu in itertools.product(lr_values, mu_values):
        print(f"Testing hyperparameters: lr={lr}, mu={mu}")

        fold_accuracies = []

        # k-fold cross-validation, i is validation, others are training
        for i in range(len(cv_folds)):
            val_df = cv_folds[i]
            train_dfs = [cv_folds[j] for j in range(len(cv_folds)) if j != i]
            train_df = pd.concat(train_dfs)

            train_x = train_df.drop('label', axis=1).to_numpy()
            train_y = train_df['label'].to_numpy()
            val_x = val_df.drop('label', axis=1).to_numpy()
            val_y = val_df['label'].to_numpy()

            model = init_perceptron(
                variant=perceptron_variant,
                num_features=num_features,
                lr=lr,
                mu=mu
            )

            model.train(x=train_x, y=train_y, epochs=epochs)

            preds = model.predict(val_x)
            fold_acc = accuracy(labels=val_y, predictions=preds)
            fold_accuracies.append(fold_acc)

            print(f"  Fold {i+1}/{len(cv_folds)}: accuracy={fold_acc:.3f}")

        # compute mean accuracy for this combination of learning rate and mu
        avg_acc = np.mean(fold_accuracies)
        print(f"  Avg accuracy for lr={lr}, mu={mu}: {avg_acc:.3f}\n")

        # update best if improved
        if avg_acc > best_avg_accuracy:
            best_avg_accuracy = avg_acc
            best_hyperparams = {'lr': lr, 'mu': mu}


    return best_hyperparams, best_avg_accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cross-validation for different hyperparameters')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=PERCEPTRON_VARIANTS, 
        help=f'Which perceptron model to run. Must be one of {PERCEPTRON_VARIANTS}.')
    parser.add_argument('--lr_values', nargs='+', type=float, default=[1], 
        help='A list (space separated) of learning rate (eta) values to try. This is the same as the initial learning rate.')
    parser.add_argument('--mu_values', nargs='+', type=float, default=[0], 
        help='A list (space separated) of margin (mu) values to try. Defaults to [0].')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='How many epochs to train for. Defaults to 10.')
    args = parser.parse_args()

    # load data
    print('load data')
    data_dict = load_data()
    cv_folds = data_dict['cv_folds']

    # run cross_validation
    print(f'run cross-validation')
    best_hyperparams, best_accuracy = cross_validation(
        cv_folds=cv_folds, 
        perceptron_variant=args.model,
        lr_values=args.lr_values, 
        mu_values=args.mu_values, 
        epochs=args.epochs)
    
    # print best hyperparameters and accuracy
    print('\nbest hyperparameters from cross-validation:\n')
    for name, value in best_hyperparams.items():
        print(f'{name:>15}: {value}')
    print(f'\n       accuracy: {best_accuracy:.3f}\n')
