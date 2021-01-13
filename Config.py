
import os

class Config:

    @staticmethod
    def car_width_base():
        return 15
    @classmethod
    def car_length_base(cls):
        return 2 * cls.car_width_base()

    @classmethod
    def path_width(cls):
        return int(3 * cls.car_width_base())

    @classmethod
    def map_size(cls):
        return cls.map_scaler() * cls.map_base()

    @staticmethod
    def map_scaler():
        return 3
    @staticmethod
    def map_base():
        return 256

    @classmethod
    def grid_size(cls):
        return cls.map_size() // cls.path_width()

    @classmethod
    def used_map_size(cls):
        return cls.path_width() * cls.grid_size()

    @staticmethod
    def result_dir():
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'res')

    @staticmethod
    def start_pos():
        return (100, 30)