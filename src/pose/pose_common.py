from enum import Enum
import numpy as np

PART_THRESHOLD = 0.01

class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]

CocoPairsIds = [(CocoPart(pair[0]), CocoPart(pair[1])) for pair in CocoPairs]

CocoPairsRender = CocoPairs[:-2]

def _round(v):
    return int(round(v))

def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None

class MissingPoseJoint(Exception):
    pass

class PosePart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()

class HumanPose:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score', 'img_w', 'img_h')

    def __init__(self, pairs):
        self.img_w = 0
        self.img_h = 0
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = PosePart(HumanPose._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = PosePart(HumanPose._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(HumanPose._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(HumanPose._get_uidx(pair.part_idx2, pair.idx2))

    def set_img_size(self, w, h):
        self.img_w = w
        self.img_h = h

    def point(self, part_enum):
        if part_enum.value not in self.body_parts:
            raise MissingPoseJoint(f'{part_enum.name}')
        p = self.body_parts[part_enum.value]
        center = (int(p.x * self.img_w + 0.5), int(p.y * self.img_h + 0.5))
        return np.array(center)

    def adjust_point(self, part_enum, p):
        if part_enum.value not in self.body_parts:
            raise MissingPoseJoint(f'{part_enum.name}')
        x = p[0] / float(self.img_w)
        y = p[1] / float(self.img_h)
        self.body_parts[part_enum.value].x = x
        self.body_parts[part_enum.value].y = y

    def midhip(self):
        if self.is_valid(CocoPart.LHip) and self.is_valid(CocoPart.RHip):
            return 0.5*(self.point(CocoPart.LHip) + self.point(CocoPart.RHip)).astype(np.int)
        elif self.is_valid(CocoPart.LHip):
            return self.point(CocoPart.LHip)
        elif self.is_valid(CocoPart.RHip):
            return self.point(CocoPart.RHip)
        else:
            raise MissingPoseJoint(f'MidHip')

    def part_exist(self, part_enum):
        return part_enum.value in self.body_parts

    def pair_length(self, part_enum_0, part_enum_1):
        return np.linalg.norm(self.point(part_enum_0) - self.point(part_enum_1))

    def is_valid(self, part_enum):
        return part_enum.value in self.body_parts

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()
