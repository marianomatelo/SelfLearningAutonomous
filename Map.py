from Utils import *
import numpy as np
from Config import Config


class Map:

    def __init__(self):
        self.margin = 10
        self.grid_size = Config.grid_size()

        self.collider_lines, self.path_rects, self.wall_rects = None, None, None
        self.bot_wall, self.right_wall, self.top_wall, self.left_wall = None, None, None, None

        self.generate_map()

    def map_frame_generator(self):

        board = np.array([
                         [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
                         [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
                         [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
                         [1.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
                         [1.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [1.,0.,1.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.],
                         [1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
                         [1.,0.,1.,0.,0.,0.,1.,0.,0.,1.,1.,1.,1.,0.,0.,1.,0.],
                         [1.,0.,1.,1.,1.,1.,1.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.],
                         [1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.],
                         [1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.],
                         [1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.],
                         [1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.],
                         [1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.],
                         [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,1.,1.,1.,1.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        ])

        return board

    def get_tile_rects(self, map_frame, tile_number):
        res = []
        for row in range(len(map_frame)):
            walls = np.where(tile_number == map_frame[row])
            walls = MiscUtils.merge_neighbors(walls[0])
            for w in walls:
                res.append(ColliderUtils.generate_block_vertice(row, row, w[0], w[1]))
        return res

    def get_path_rect(self, map_frame):
        return self.get_tile_rects(map_frame, 1)

    def get_wall_rect(self, map_frame):
        return self.get_tile_rects(map_frame, 0)

    def draw_blocks(self, map, blocks, draw_method):
        for b in blocks:
            draw_method(map, b)

    def get_boundaries(self):
        bottom_wall = (
            (0, self.grid_size * Config.path_width()),
            (0, Config.map_size()),
            (Config.map_size(), Config.map_size()),
            (Config.map_size(), self.grid_size * Config.path_width())
        )
        right_wall = (
            (self.grid_size * Config.path_width(), 0),
            (self.grid_size * Config.path_width(), Config.map_size()),
            (Config.map_size(), Config.map_size()),
            (Config.map_size(), 0)
        )
        top_wall = (
            (0,0),
            (0,0),
            (self.grid_size * Config.path_width(), 0),
            (self.grid_size * Config.path_width(), 0)
        )
        left_wall = (
            (0, 0),
            (0, self.grid_size * Config.path_width()),
            (0, self.grid_size * Config.path_width()),
            (0, 0)
        )
        return bottom_wall, right_wall, top_wall, left_wall


    def generate_map(self):

        map_frame = self.map_frame_generator()

        self.path_rects = self.get_path_rect(map_frame)

        self.wall_rects = self.get_wall_rect(map_frame)

        self.bot_wall, self.right_wall, self.top_wall, self.left_wall = self.get_boundaries()

        self.collider_lines = ColliderUtils.collider_lines_from_path_rects(self.path_rects)


    def draw_map_bg(self):

        map_image = ImageUtils.draw_map()
        self.draw_blocks(map_image, self.path_rects, ImageUtils.draw_path)
        self.draw_blocks(map_image, self.wall_rects, ImageUtils.draw_wall)
        ImageUtils.draw_rect(map_image, self.bot_wall, color='white')
        ImageUtils.draw_rect(map_image, self.right_wall, color='white')


        for l in self.collider_lines:
            ImageDraw.Draw(map_image).line([tuple(l[0]), tuple(l[1])], fill=(0,0,0), width=2)
        return map_image