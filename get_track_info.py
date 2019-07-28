import requests
import os
import re
import base64
import pandas as pd
from tqdm import tqdm
import sys
import json

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


# TODO: flatten response for storage in pandas dataframe
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
    results['available_markets'] = response['available_markets']
    results['duration_ms'] = response['duration_ms']
    results['name'] = response['name']
    results['popularity'] = response['popularity']
    results['uri'] = response['uri']

    for artist in response['album']['artists']:
        artist.pop('uri')
        artist.pop('external_urls')
        artist.pop('type')
        artist.pop('href')

    results['album'] = response['album']

    results['artists'] = list()

    for artist in response['artists']:
        results['artists'].append({
            'name': artist['name'],
            'id': artist['id']
        })

    return results


if __name__=="__main__":
    df = pd.read_pickle('track_ids.pkl')
    df = df.set_index('track_id')

    final_df = pd.DataFrame()

    for i, row in df.iterrows():
        current_dict = {'track_id': i, 'genre': row['genre']}
        track_info = get_track(i)
        track_features = get_track_audio_features(i)
        # track_analysis = get_track_audio_analysis(i)

        x = {**current_dict, **track_info}
        z = {**x, **track_features}
        
        final_df.append(z)

        print(final_df)

        exit(1)
