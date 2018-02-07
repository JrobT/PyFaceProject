import spotipy
import pprint
import spotipy.util as util


class MusicManager:
    """Class for music player build."""

    def __init__(self, username):
        """Initialise the music manager with account information."""
        self.username = username
        self.scope = 'playlist-modify-public'
        self.client_id = 'cb4ab90932be46d0848653c27469f61c'
        self.client_secret = 'e9bd8cc2fca547d99cb05ca49472e2d2'
        self.token = util.prompt_for_user_token(username, self.scope,
                                                self.client_id,
                                                self.client_secret)

        if self.token:
            self.sp = spotipy.Spotify(auth=self.token)
            self.sp.trace = False

        search_str = 'Muse'
        result = self.sp.search(search_str)
        pprint.pprint(result)
        # TODO : Read .csv for track info
        # TODO : Load track info into data

    def create_playlist(self, desc, emotion):
        """Create a playlist for the user based on the emotion."""
        if self.token:
            playlists = self.sp.user_playlist_create(self.username,
                                                     "PyFace Playlist", desc)
            # TODO : Test creating playlist
            # TODO : Get playlist_id
        else:
            print("Can't get token for {}".format(self.username))

    def add_music():
        """Replace the music in 'PyFace Playlist' with the new emotion music."""
        scope = 'playlist-modify-public'

        if self.token:
            results = self.sp.user_replace_tracks(self.username,
                                                  self.playlist_id, track_ids)
