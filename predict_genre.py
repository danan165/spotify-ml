import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn import tree


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


def convert_cols_to_features(pred_df):
    # convert columns to features (artists, available markets, album release data)
    for i, row in tqdm(pred_df.iterrows()):

        # TODO: only use number of artists

        # TODO: only use number of available markets

        # convert release date to datetime
        if row['album_release_date_precision'] == 'day':
            # yyyy-mm-dd
            pred_df.loc[i, 'album_release_date'] = datetime.strptime(row['album_release_date'], '%Y-%m-%d')
        elif row['album_release_date_precision'] == 'year':
            # yyyy
            pred_df.loc[i, 'album_release_date'] = datetime.strptime(row['album_release_date'], '%Y')
        else:
            # yyyy-mm
            pred_df.loc[i, 'album_release_date'] = datetime.strptime(row['album_release_date'], '%Y-%m')

    return pred_df


def drop_non_feature_columns(pred_df):
    # drop non-feature columns
    cols_to_drop = ['album_artists', 'album_id', 'album_name', 'artists',
    'album_uri', 'name', 'track_id', 'uri', 'available_markets', 'album_release_date_precision']
    pred_df = pred_df.drop(columns=cols_to_drop)
    return pred_df


if __name__=="__main__":

    # load dataset
    df = pd.read_pickle('predict_genre_dataset.pkl')

    # clean dataset for training/testing
    df = drop_duplicate_tracks(df)
    df = convert_cols_to_features(df)
    df = drop_non_feature_columns(df)

    # split into training and testing set

    # train the model

    # evaluate the model (precision and recall)