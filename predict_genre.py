import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics


def drop_duplicate_tracks(df):
    # examine dataset makeup
    # print('shape of original dataframe: ', df.shape)
    # print('number of unique track ids in original dataframe: ', len(df.track_id.unique()))
    # print('number of duplicated track id rows in original dataframe: ', df[df.duplicated(subset='track_id')].shape)

    # drop duplicates
    df.sort_values('track_id', inplace=True)
    df.drop_duplicates(subset='track_id', keep='first', inplace=True)
    # print(df.shape)
    # print(len(df.track_id.unique()))

    # unique tracks per genre
    # pop = df.loc[df['genre'] == 'pop']
    # hip_hop = df.loc[df['genre'] == 'hip-hop']
    # rock_n_roll = df.loc[df['genre'] == 'rock-n-roll']

    # print(len(pop.track_id.unique()))           # 892
    # print(len(hip_hop.track_id.unique()))       # 905
    # print(len(rock_n_roll.track_id.unique()))   # 990

    return df

# convert columns to features (artists, available markets, album release data)
def convert_cols_to_features(pred_df):
    for i, row in tqdm(pred_df.iterrows()):

        # TODO: only use number of artists

        # TODO: only use number of available markets

        # convert release date to timestamp
        if row['album_release_date_precision'] == 'day':
            # yyyy-mm-dd
            date_time = datetime.strptime(row['album_release_date'], '%Y-%m-%d')
            pred_df.loc[i, 'album_release_date'] = date_time.timestamp()
        elif row['album_release_date_precision'] == 'year':
            # yyyy
            date_time = datetime.strptime(row['album_release_date'], '%Y')
            pred_df.loc[i, 'album_release_date'] = date_time.timestamp()
        elif row['album_release_date_precision'] == 'month':
            # yyyy-mm
            date_time = datetime.strptime(row['album_release_date'], '%Y-%m')
            pred_df.loc[i, 'album_release_date'] = date_time.timestamp()

        # album type must be mapped to ints
        if row['album_type'] == 'single':
            pred_df.loc[i, 'album_type'] = 0
        elif row['album_type'] == 'album':
            pred_df.loc[i, 'album_type'] = 1
        elif row['album_type'] == 'compilation':
            pred_df.loc[i, 'album_type'] = 2
        

    return pred_df


def drop_non_feature_columns(pred_df):
    # drop non-feature columns
    cols_to_drop = ['track_id', 'album_release_date_precision', 'available_markets', \
                    'album_id', 'album_artists', 'uri', 'album_name', 'album_uri', 'artists', \
                    'name']
    pred_df = pred_df.drop(columns=cols_to_drop)
    return pred_df


def plot_feature_importance(model, df):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    #sort feature names for display
    names = [df.columns[i] for i in indices]

    # Create plot
    plt.figure()

    # Create plot title
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(df.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(df.shape[1]), names, rotation=90)

    # Save plot
    plt.savefig(fname=os.path.join(os.path.dirname(__file__), 'feature-importance', 'feature_importance.png'), bbox_inches='tight')


def print_precision_scores(y_test, y_predict):
    print('Macro Precision: ', precision_score(y_test, y_predict, average='macro'))
    print('Micro Precision: ', precision_score(y_test, y_predict, average='micro'))
    print('Weighted Precision: ', precision_score(y_test, y_predict, average='weighted'))


def print_recall_scores(y_test, y_predict):
    print('Macro Recall: ', recall_score(y_test, y_predict, average='macro'))
    print('Micro Recall: ', recall_score(y_test, y_predict, average='micro'))
    print('Weighted Recall: ', recall_score(y_test, y_predict, average='weighted'))


def decision_tree_prediction(X_train, y_train, X_test, y_test):
    # create the model
    print('creating the model...')
    model = tree.DecisionTreeClassifier()

    # fit the model
    print('training the model...')
    model.fit(X_train, y_train)

    # evaluate the model (precision and recall)
    print('testing the model...')
    y_predict = model.predict(X_test)

    # classification report
    print('saving classification report...')
    target_names = ['pop', 'hip-hop', 'rock-n-roll']
    report = classification_report(y_test, y_predict, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(r'model-performance/classification_report.csv')
    
    # feature importance
    plot_feature_importance(model, X)

    # precision/recall scores
    print_precision_scores(y_test, y_predict)
    print_recall_scores(y_test, y_predict)


# TODO: fix performance for multi-label classification!!!!
def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """

    score = 0

    confusion_mtx = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])
    tn = confusion_mtx[1][1]
    tp = confusion_mtx[0][0]
    fn = confusion_mtx[0][1] # row 0, col 1
    fp = confusion_mtx[1][0] # row 1, col 0

    if metric=='accuracy':
        # score = (tp + tn)/(tp + tn + fp + fn)
        score = metrics.accuracy_score(y_true, y_pred)
    elif metric=='f1-score':
        score = (2*tp)/((2*tp) + fp + fn)
    elif metric=='precision':
        score = (tp)/(tp + fp)
    elif metric=='sensitivity':
        score = (tp)/(tp + fn)
    elif metric=='specificity':
        score = (tn)/(tn + fp)

    # print('new score: ', score, '\n')
    return score


def adaboost_cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of an adaboost classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """

    print('k fold cv to eval performance...')

    #Put the performance of the model on each fold in the scores array
    scores = []

    skf = StratifiedKFold(n_splits=k)
    skf.get_n_splits(X, y)
    StratifiedKFold(n_splits=k, random_state=None, shuffle=False)

     # calculate accuracy performance of the model with each fold
    for train_index, test_index in skf.split(X, y):
        print('new fold...')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        perf = performance(y_test, y_pred, metric=metric)
        scores.append(perf)
       
    #And return the average performance across all fold splits.
    return np.array(scores).mean()


def select_adaboost_model_with_params(X, y, k=5, metric='accuracy'):
    learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    n_estimators = [25, 50, 100, 150]

    max_pred_score = 0
    best_clf = None

    # hyperparameter grid search
    print('hyperparameter grid search...\n')
    for lr in learning_rates:
        for n in n_estimators:
            clf = AdaBoostClassifier(n_estimators=n, learning_rate=lr)
            print('training adaboost for lr=', lr, ', n_estimators=', n)

            avg_performance = adaboost_cv_performance(clf, X, y, metric=metric)
            
            if avg_performance > max_pred_score:
                max_pred_score = avg_performance
                best_clf = clf
                print('new high score: ', max_pred_score)
                print('winning params: lr=', lr, ', n=', n)

    return best_clf


def print_adaboost_performance(X, y):
    clf = select_adaboost_model_with_params(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    clf.fit(X_train, y_train)

    metrics = ['accuracy', 'f1-score', 'precision', 'sensitivity', 'specificity']

    y_pred = clf.predict(X_test)

    for metric in metrics:
        perf = performance(y_test, y_pred, metric=metric)
        print(metric, ': ', perf)
    

if __name__=="__main__":

    # load dataset
    df = pd.read_pickle('datasets/predict_genre_dataset.pkl')

    # clean dataset for training/testing
    print('cleaning the dataset...')
    df = drop_duplicate_tracks(df)
    df = convert_cols_to_features(df)
    df = drop_non_feature_columns(df)

    # split into training and testing set
    print('splitting into training and testing...')
    X = df.drop('genre', axis=1)    # features w/o labels
    y = df['genre']                 # labels

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    X_numpy_array = X.values
    y_numpy_array = y.values

    # make prediction
    print_adaboost_performance(X_numpy_array, y_numpy_array)
    
