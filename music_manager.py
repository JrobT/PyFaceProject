"""."""

import spotipy
import requests
import json
import base64
import csv
import spotipy.util as util


class MusicManager:
    """Class for music player build."""

    def __init__(self, username):
        """Initialise the music manager with account information."""
        self.username = username
        self.scope = 'playlist-modify-public playlist-modify-private playlist-read-private'
        self.client_id = 'cb4ab90932be46d0848653c27469f61c'
        self.client_secret = 'e9bd8cc2fca547d99cb05ca49472e2d2'
        self.redirect_uri = "https://example.com/callback/"
        self.token = util.prompt_for_user_token(self.username, self.scope,
                                                self.client_id,
                                                self.client_secret,
                                                self.redirect_uri)

        if self.token:
            self.sp = spotipy.Spotify(auth=self.token)
            self.sp.trace = False

    def request_token(self):
        """."""
        auth_url = 'https://accounts.spotify.com/authorize'
        payload = {'client_id': self.client_id,
                   'response_type': 'code',
                   'redirect_uri': self.redirect_uri}
        auth = requests.get(auth_url, data=json.dumps(payload))
        print(auth)

        # resp_url = input('\nThen please copy-paste the url you where redirected to: ')
        # resp_code = resp_url.split("?code=")[1].split("&")[0]

        token_url = 'https://accounts.spotify.com/api/token'
        payload = {'redirect_uri': self.redirect_uri,
                   # 'code': resp_code,
                   'grant_type': 'authorization_code',
                   'scope': self.scope}
        comb = '{}:{}'.format(self.client_id, self.client_secret).encode('ascii')
        auth_header = base64.b64encode(comb)
        headers = {'Authorization': 'Basic {}'.format(auth_header)}
        req = requests.post(token_url, data=json.dumps(payload),
                            headers=headers, verify=True)
        print(req)
        response = req.json()

        return response

    def get_track_ids(emotion):
        """Read 'songs.csv' and retrieve track ids."""
        track_ids = []
        with open('songs.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                track_ids.append('spotify:track:' + row[emotion])
        return track_ids

    def create_playlist(self, desc, emotion):
        """Create a playlist for the user based on the emotion."""
        token = self.request_token()
        print(token)
        access_token = token['access_token']
        auth_header = {'Authorization': 'Bearer {token}'
                       .format(token=access_token),
                       'Content-Type': 'application/json'}
        api_url = 'https://api.spotify.com/v1/users/{}/playlists'.format(self.username)
        payload = {'name': "PyFace Playlist", 'public': 'false'}
        r = requests.post(api_url, data=json.dumps(payload), headers=auth_header)

        # if self.token:
        #     playlists = self.sp.user_playlist_create(self.username,
        #                                              "PyFace Playlist",
        #                                              desc)
        # playlists = self.sp.user_playlists(self.username)
        # for playlist in playlists['items']:
        #     if "PyFace Playlist" == playlist['name']:
        #         playlist_id = playlist['playlist_id']
        # track_ids = self.get_track_ids(emotion)
        # results = self.sp.user_playlist_replace_tracks(self.username,
        #                                                playlist_id,
        #                                                track_ids)
        # else:
        #     print("Can't get token for {}".format(self.username))
