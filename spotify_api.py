import requests
import os
import re

"""
run these commands to refresh the token
$ cd web-api-auth-examples/client_credentials
$ node app.js
"""
token = 'BQB_rAFxuDjmJ01zBXu-AOYbgdkgoQCdWmVaz8XLrtlaaznNGZolxfpLcKBuc_JyninYkhzkpEuYWuLyOdg'


def get_available_genre_seeds():
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


def get_recommendations_by_seed_track(seed_track_id):
    url = 'https://api.spotify.com/v1/recommendations?seed_tracks={}&limit=50'.format(seed_track_id)

    headers = {
        'Authorization': 'Bearer ' + token
    }

    r = requests.get(url = url, headers = headers)

    data = r.json()

    print(data)


def get_track_audio_features(track_id):
    url = 'https://api.spotify.com/v1/audio-features/{}'.format(track_id)

    headers = {
        'Authorization': 'Bearer ' + token
    }

    r = requests.get(url = url, headers = headers)

    data = r.json()

    print(data)


def get_track_audio_analysis(track_id):

    url = 'https://api.spotify.com/v1/audio-analysis/{}'.format(track_id)

    headers = {
        'Authorization': 'Bearer ' + token
    }

    r = requests.get(url = url, headers = headers)

    data = r.json()

    print(data)


if __name__=='__main__':
    # seed_track_ids = list()

    # with open("seed_tracks.txt", "r") as f:
    #     content = f.readlines()

    # for line in content:
    #     seed_track_ids.append(re.search('\:(.*)', line).group()[2:])

    seed_genres = get_available_genre_seeds()

    list_1 = ['pop', 'new-release', 'romance']

    list_2 = ['tango', 'dance']

    list_3 = ['edm', 'club']

    list_4 = ['drum-and-bass', 'jazz', 'funk']

    # print('SEED GENRES: ')

    # print(seed_genres)

    track_ids = get_track_recommendations_by_seed_genres(list_4)

    # print('\n\nTRACK IDS: ')

    # print(track_ids)

    get_track_audio_features(track_ids[0])

