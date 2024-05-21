from .adviser import Adviser
from environment.utils import *
import math

class AdviserWordKeeper(Adviser):
    '''
    도착 지역의 나라의 effective_ratio를 가치에 곱해줌
    '''
    def __init__(self, static_dict, me, weight):
        super().__init__(static_dict, me)
        self.operator = self.multiply
        if weight < -1:
            raise Exception('Word Keeper\'s weight ERROR')
        self.weight = weight

    def evaluate(self, advise, obs, infos):
        region = self.get_final_destination(advise, obs['loc2power'])
        effective_ratio = infos["effective_ratio"]

        if region != None:
            victim = obs["loc2power"][region]
            if victim != None and victim != self.me:
                return (effective_ratio[victim] ** (self.weight + 1)).real

        return 1.0
