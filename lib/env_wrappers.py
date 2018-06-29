import sys
import os

from vizdoom import DoomGame, ScreenResolution

CONFIGURATIONS_DIR = os.path.join(
    os.environ['CONDA_PREFIX'],
    "lib/python" + str(sys.version_info.major) + "." + str(sys.version_info.minor) + "/site-packages/vizdoom/scenarios")
CONFIGURATIONS = {  # https://github.com/mwydmuch/ViZDoom/tree/master/scenarios:
    "DoomTakeCover": "take_cover"
}


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ViZDoomWrapper(object):
    def __init__(self, configuration):
        configuration = CONFIGURATIONS[configuration]
        game = DoomGame()
        game.load_config(
            os.path.join(CONFIGURATIONS_DIR, configuration + ".cfg"))
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_window_visible(False)
        game.init()
        action_dim = game.get_available_buttons_size()
        action_space = AttrDict()
        action_space.low = [0 for i in range(action_dim)]
        action_space.high = [1 for i in range(action_dim)]
        self.action_space = action_space
        self.game = game

    def reset(self):
        self.game.new_episode()
        return self.game.get_state().screen_buffer

    def step(self, action):
        action = action.astype(bool).tolist()
        reward = self.game.make_action(action)
        if self.game.get_state() is not None:
            self.last_screen_buffer = self.game.get_state().screen_buffer
        return self.last_screen_buffer, \
               reward, \
               self.game.is_episode_finished(), \
               None

    def close(self):
        self.game.close()
