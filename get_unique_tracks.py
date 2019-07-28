import requests
import os
import re
import base64
import pandas as pd
from tqdm import tqdm
import sys

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


def get_available_genre_seeds():
    token = get_token()

    url = 'https://api.spotify.com/v1/recommendations/available-genre-seeds'

    headers = {
        'Authorization': 'Bearer ' + token
    }

    r = requests.get(url = url, headers = headers)

    data = r.json()

    genres_list = list()

    for genre in data['genres']:
        genres_list.append(str(genre))

    return genres_list


def get_track_recommendations_by_seed_genres(seed_genres):
    token = get_token()

    seed_genres_param = ''

    for genre in seed_genres:
        seed_genres_param = seed_genres_param + str(genre) + ','

    seed_genres_param = seed_genres_param[:-1]

    url = 'https://api.spotify.com/v1/recommendations?seed_genres={}&limit=100'.format(seed_genres_param)

    headers = {
        'Authorization': 'Bearer ' + token
    }

    r = requests.get(url = url, headers = headers)

    data = r.json()

    track_ids = list()

    for track in data['tracks']:
        track_ids.append(str(track['id']))
        # if track['preview_url'] is not None:
        #     print(track['name'], track['preview_url'])

    return track_ids


def get_track_recommendations_by_seed_genre(seed_genre):
    token = get_token()

    url = 'https://api.spotify.com/v1/recommendations?seed_genres={}&limit=100'.format(seed_genre)

    headers = {
        'Authorization': 'Bearer ' + token
    }

    r = requests.get(url = url, headers = headers)

    data = r.json()

    track_ids = list()

    for track in data['tracks']:
        track_ids.append(str(track['id']))

    return track_ids


if __name__=='__main__':

    # Create dataset as a dataframe
    df = pd.DataFrame(columns = ['track_id', 'genre'])

    # use "get_available_genre_seeds" function to pick some seed genres.
    seed_genres = ['pop', 'hip-hop', 'rock-n-roll']

    track_ids = set()

    repetitions = 700

    for genre in tqdm(seed_genres):
        i = 0
        while i < repetitions:
            track_ids.update(get_track_recommendations_by_seed_genre(genre))
            i += 1

        df2 = pd.DataFrame({'track_id': list(track_ids)})
        df2['genre'] = genre
        df = df.append(df2, ignore_index=True)
        track_ids = set()

    print(df.shape)

    df.to_pickle('track_ids_copy.pkl')

