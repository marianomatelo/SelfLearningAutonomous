
from Map import Map
from reinforcement_trainer import ReinforcementTrainer


class Game:

    def __init__(self):
        self.map = Map()
        self.colliders = self.map.collider_lines
        self.wall_rects = self.map.wall_rects
        self.result_file = '.gif'
        self.best = 0

    def run_reinfocement(self):
        trainer = ReinforcementTrainer(self.map, 300000000000)
        trainer.train()


if __name__ == "__main__":

    Game().run_reinfocement()

