# spotify-ml

Using supervised machine learning to classify a song's genre as pop, hip hop, or rock-n-roll based on properties returned from the Spotify API.

## Dataset Creation

Currently a two step process: retrieving a random list of tracks from each genre, then retrieving properties for each track.

### Retrieve a List of Tracks

To retrieve a random list of tracks per genre that would be sufficiently large enough for training/testing the model, I used the following endpoint from the spotify API: [Get Recommendations from Seed Genre](https://developer.spotify.com/documentation/web-api/reference/browse/get-recommendations/).

I iteratively queried this endpoint and added track ids to a set in order to retrieve a large, random set of ids with no duplicates. The endpoint can only retrieve a maximum of 100 tracks at a time.

All of this work is done in the `get_unique_tracks.py` file under the `dataset-creation` folder.

Running this file will result in a saved pickle file (`datasets/track_ids.pkl`) that contains a pandas dataframe with two columns: track_id, genre.

### Retrieve Properties for Each Unique Track

I retrieved properties of each track by querying the following spotify API endpoints: [Get a Track](https://developer.spotify.com/documentation/web-api/reference/tracks/get-track/), [Get Audio Features for a Track](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/).

I added each property as a column in a pandas dataframe. All of this work is done in `dataset-creation/get_track_info.py`.

Running this file will result in a saved pickle file (`datasets/predict_genre_dataset.pkl`) that contains a dataframe where each row is a track, and each column is a property of that track.

## Feature Creation / Model Creation

This is done in `predict_genre.py`.