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


def get_track_audio_features(track_id):
    token = get_token()

    url = 'https://api.spotify.com/v1/audio-features/{}'.format(track_id) 

    response = requests.get(
        url = url, 
        headers = {'Authorization': 'Bearer ' + token}
    ).json()

    response.pop('id')
    response.pop('uri')
    response.pop('track_href')
    response.pop('analysis_url')
    response.pop('type')
    response.pop('duration_ms')

    return response


def get_track_audio_analysis(track_id):
    token = get_token()

    url = 'https://api.spotify.com/v1/audio-analysis/{}'.format(track_id)

    response = requests.get(
        url = url, 
        headers = {'Authorization': 'Bearer ' + token}
    ).json()

    return response


def get_track(track_id):
    token = get_token()

    url = 'https://api.spotify.com/v1/tracks/{}'.format(track_id)

    response = requests.get(
        url = url, 
        headers = {'Authorization': 'Bearer ' + token}
    ).json()

    response['album'].pop('images')
    response['album'].pop('href')
    response['album'].pop('available_markets')
    response['album'].pop('external_urls')

    results = {}

    results['explicit'] = response['explicit']
    #results['available_markets'] = response['available_markets']
    results['duration_ms'] = response['duration_ms']
    results['name'] = response['name']
    results['popularity'] = response['popularity']
    results['uri'] = response['uri']

    for artist in response['album']['artists']:
        artist.pop('uri')
        artist.pop('external_urls')
        artist.pop('type')
        artist.pop('href')

    #results['album_artists'] = response['album']['artists']
    results['album_total_tracks'] = response['album']['total_tracks']
    results['album_name'] = response['album']['name']
    results['album_uri'] = response['album']['uri']
    results['album_release_date'] = response['album']['release_date']
    results['album_release_date_precision'] = response['album']['release_date_precision']
    results['album_id'] = response['album']['id']
    results['album_type'] = response['album']['album_type']

    results['artists'] = list()

    for artist in response['artists']:
        results['artists'].append({
            'name': artist['name'],
            'id': artist['id']
        })

    return results


if __name__=="__main__":
    datasets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    df = pd.read_pickle(os.path.join(datasets_path, 'track_ids.pkl'))
    df = df.set_index('track_id')

    final_df_keys = ['album_release_date_precision', 'valence', 
    'popularity', 'album_id', 'speechiness', 'album_artists', 'loudness', 
    'album_uri', 'available_markets', 'name', 'artists', 'duration_ms', 'track_id', 
    'key', 'album_total_tracks', 'explicit', 'album_release_date', 'mode', 'album_name', 
    'instrumentalness', 'liveness', 'tempo', 'acousticness', 'genre', 'danceability', 
    'album_type', 'uri', 'energy', 'time_signature']

    final_df = pd.DataFrame(columns=final_df_keys)

    #df  = pd.DataFrame([podcast_dict], columns=podcast_dict.keys())
    #df_podcast = pd.concat([df_podcast, df], axis =0).reset_index()

    pp = pprint.PrettyPrinter(indent=4)

    for i, row in tqdm(df.iterrows()):
        current_dict = {'track_id': i, 'genre': row['genre']}
        track_info = get_track(i)
        track_features = get_track_audio_features(i)
        # track_analysis = get_track_audio_analysis(i)

        x = {**current_dict, **track_info}
        z = {**x, **track_features}

        #pp.pprint(z)

        temp_df = pd.DataFrame(z)
        final_df = pd.concat([final_df, temp_df], axis=0).reset_index(drop=True)
        final_df.set_index('track_id')
        final_df.to_pickle(os.path.join(datasets_path, 'predict_genre_dataset_copy.pkl'))

        #print(tabulate(final_df, headers='keys', tablefmt='psql'))
