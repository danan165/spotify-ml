import requests
import os
from pathlib import Path
import re
import base64
import pandas as pd
from tqdm import tqdm
import sys
import json
import pprint
from tabulate import tabulate

def get_token():
    # "credentials.txt" is a two line file that contains your
    # client id and your client secret.

    with open("credentials.txt", "r") as f:
        content = f.readlines()

    client_id = content[0].strip()
    client_secret = content[1].strip()

    encoded = base64.b64encode((client_id + ':' + client_secret).encode('utf-8'))

    url = 'https://accounts.spotify.com/api/token'

    headers = {
        'Authorization': 'Basic ' + encoded.decode('utf-8')
    }

    data = {
        'grant_type': 'client_credentials'
    }

    r = requests.post(url = url, headers = headers, data = data)

    response = r.json()
    return response['access_token']


def get_track_audio_features(track_id, df_keys):
    token = get_token()

    url = 'https://api.spotify.com/v1/audio-features/{}'.format(track_id) 

    response = requests.get(
        url = url, 
        headers = {'Authorization': 'Bearer ' + token}
    ).json()

    results = {}

    for df_key in df_keys:
        if df_key in response.keys():
            results[df_key] = response[df_key]

    return results


def get_track_audio_analysis(track_id, df_keys):
    token = get_token()

    url = 'https://api.spotify.com/v1/audio-analysis/{}'.format(track_id)

    response = requests.get(
        url = url, 
        headers = {'Authorization': 'Bearer ' + token}
    ).json()

    results = {}

    for df_key in df_keys:
        if df_key in response.keys():
            results[df_key] = len(response[df_key])

    return results


def get_track(track_id, df_keys):
    token = get_token()

    url = 'https://api.spotify.com/v1/tracks/{}'.format(track_id)

    response = requests.get(
        url = url, 
        headers = {'Authorization': 'Bearer ' + token}
    ).json()

    results = {}

    for df_key in df_keys:
        if df_key in response.keys():
            if df_key == 'available_markets':
                results[df_key] = len(response[df_key])
            else:
                results[df_key] = response[df_key]
        elif 'album' in response.keys():
            if df_key == 'album_release_date':
                results[df_key] = response['album']['release_date']
            elif df_key == 'album_release_date_precision':
                results[df_key] = response['album']['release_date_precision']
            elif df_key == 'album_total_tracks':
                results[df_key] = response['album']['total_tracks']
            elif df_key == 'album_type':
                results[df_key] = response['album']['album_type']

    return results


if __name__=="__main__":
    datasets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    df = pd.read_pickle(os.path.join(datasets_path, 'track_ids_tpr.pkl'))
    df = df.set_index('track_id')

    final_df_keys = list()

    with open(os.path.join(datasets_path, 'df_keys.txt')) as f:
        for line in f.readlines():
            final_df_keys.append(line.strip())

    final_df = pd.DataFrame(columns=final_df_keys)

    pp = pprint.PrettyPrinter(indent=4)

    for i, row in tqdm(df.iterrows()): 
        current_dict = {'track_id': i, 'genre': row['genre']}
        track_info = get_track(i, final_df_keys)
        track_features = get_track_audio_features(i, final_df_keys)
        track_analysis = get_track_audio_analysis(i, final_df_keys)

        x = {**current_dict, **track_info}
        y = {**x, **track_features}
        z = {**y, **track_analysis}

        # pp.pprint(z)

        temp_df = pd.DataFrame.from_records(z, columns=final_df_keys, index=['track_id'])
        final_df = pd.concat([final_df, temp_df], axis=0).reset_index(drop=True)
        final_df.set_index('track_id')

        #print(tabulate(final_df, headers='keys', tablefmt='psql'))

    final_df.to_pickle(os.path.join(datasets_path, 'predict_genre_dataset_tpr.pkl'))
