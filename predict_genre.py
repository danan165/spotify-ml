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
    cols_to_drop = ['track_id', 'album_release_date_precision']
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


if __name__=="__main__":

    # load dataset
    df = pd.read_pickle('datasets/predict_genre_dataset_copy.pkl')

    # clean dataset for training/testing
    print('cleaning the dataset...')
    df = drop_duplicate_tracks(df)
    df = convert_cols_to_features(df)
    df = drop_non_feature_columns(df)

    # split into training and testing set
    print('splitting into training and testing...')
    X = df.drop('genre', axis=1)    # features w/o labels
    y = df['genre']                 # labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

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
    
