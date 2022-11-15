from Tables import *


# 棋盘和搜索参数
MIN_DEPTH = 2
MAX_DEPTH = 8
MAX_ACTION_NUM = 20
TIME_MATCH = 90
TIME_TURN = 15

# 方向
MOV = [1, 32, 33, 31]

# 1~4范围内的邻居点的偏移值
NEIGHBOR_1 = [-33, -32, -31, -1, 1, 31, 32, 33]
NEIGHBOR_2 = [-66, -64, -62, -33, -32, -31, -2, -1, 1, 2, 31, 32, 33, 62, 64, 66]  # 8*2=16个直线进攻点
NEIGHBOR_3 = [
	-33, -32, -31, -1, 1, 31, 32, 33, -66, -64, -62, -2, 2, 62, 64, 66,
	-99, -96, -93, -3, 3, 93, 96, 99, -65, -63, -34, -30, 30, 34, 63, 65
]  # 8*3=24个直线进攻点 + 8个八卦防守点
NEIGHBOR_4 = [
	-33, -32, -31, -1, 1, 31, 32, 33, -66, -64, -62, -2, 2, 62, 64, 66,
	-99, -96, -93, -3, 3, 93, 96, 99, -132, -128, -124, -4, 4, 124, 128, 132
]  # 8*4=32个直线进攻点


# 棋子（同时表示玩家）
PIECE_BLACK = 0
PIECE_WHITE = 1
PIECE_EMPTY = 2
PIECE_OUTSIDE = 3

# 打印出的棋子样式
VISUAL_PIECE_EMPTY = '.'
VISUAL_PIECE_BLACK = 'x'
VISUAL_PIECE_WHITE = 'o'
VISUAL_PIECE_BLACK_CUR = 'X'
VISUAL_PIECE_WHITE_CUR = 'O'

# 棋型得分
WIN_SCORE = 10000  # 胜利局面评分（一定是这个分）
LOSE_SCORE = -WIN_SCORE

# 哈希表标志
HASH_EXACT = 0
HASH_ALPHA = 1
HASH_BETA = 2
HASH_UNKNOWN = 3
KILL_DEPTH = 99

# 算杀参数
MAX_VCF_DEPTH = 20  # 最大vcf深度
MAX_VCT_DEPTH = 8  # 最大vct深度
MAX_VCT_TIME = 1  # VCT时间

# 单点棋型
NONE = 0
BLOCK1 = 1
FLEX1 = 2
BLOCK2 = 3
FLEX2 = 4
BLOCK3 = 5
FLEX3 = 6
BLOCK4 = 7
FLEX4 = 8
FIVE = 9

# 交叉棋型
A = 14  # 连五
B = 13  # 活四/双冲四
C = 12  # 冲四活三
D = 11  # 冲四眠三/冲四活二
E = 10  # 冲四眠二
F = 9   # 冲四
G = 8   # 双活三
H = 7   # 活三眠三/活三活二
I = 6   # 活三眠二
J = 5   # 活三
K = 4   # 眠三活二
L = 3   # 眠三
M = 2   # 活二
N = 1   # 眠二


def pointX(position: int) -> int:
	"""获得点的X坐标"""
	return position >> 5


def pointY(position: int) -> int:
	"""获得点的Y坐标"""
	return position & 31


def pointX_V(position):
	"""获得点的X坐标（可视化）"""
	return pointX(position) - 4


def pointY_V(position):
	"""获得点的Y坐标（可视化）"""
	return pointY(position) - 4


def pointXY(position):
	"""获得点的XY坐标"""
	return pointX(position), pointY(position)


def pointXY_V(position):
	"""获得点的XY坐标（可视化）"""
	return pointX_V(position), pointY_V(position)


def makePoint(x, y):
	"""从坐标生成点"""
	return (x << 5) + y


def makePoint_V(x, y):
	"""从可视化的坐标生成点"""
	return makePoint(x + 4, y + 4)


def distance(pos1, pos2):
	"""计算两点在棋盘上的距离"""
	dx = pointX(pos1) - pointX(pos2)
	dy = pointY(pos1) - pointY(pos2)
	
	if dx == 0:
		return dy if dy > 0 else -dy
	if dy == 0:
		return dx if dx > 0 else -dx
	if dx == dy or dx == -dy:
		return dx if dx > 0 else -dx
	return -1


class Chess:
	"""棋子类，棋盘的每个Cell是一个棋子类的实例"""
	
	def __init__(self):
		self.piece = PIECE_EMPTY
		self.neighbor = 0
		self.pattern = [[0 for _ in range(2)] for _ in range(4)]  # Line Key
		self.shape = [[NONE for _ in range(2)] for _ in range(4)]  # Line Shape
		self.shape4 = [0 for _ in range(2)]  # Cross Shape

	def update1(self, k):
		"""更新单线棋型"""
		pattern = self.pattern[k]
		shape = self.shape[k]
		shape[0] = SHAPE_TABLE[pattern[0]][pattern[1]]
		shape[1] = SHAPE_TABLE[pattern[1]][pattern[0]]

	def update4(self):
		"""更新四线棋型"""
		shape = self.shape
		self.shape4[0] = FOUR_SHAPE_TABLE[shape[0][0]][shape[1][0]][shape[2][0]][shape[3][0]]
		self.shape4[1] = FOUR_SHAPE_TABLE[shape[0][1]][shape[1][1]][shape[2][1]][shape[3][1]]

	def updateShape(self):
		"""更新四个单线棋型"""
		self.update1(0)
		self.update1(1)
		self.update1(2)
		self.update1(3)

	def prior(self, player):
		"""计算点的优先级"""
		pattern = self.pattern
		v1 = SHAPE_PRIOR[pattern[0][1]][pattern[0][0]] + SHAPE_PRIOR[pattern[1][1]][pattern[1][0]] +\
			 SHAPE_PRIOR[pattern[2][1]][pattern[2][0]] + SHAPE_PRIOR[pattern[3][1]][pattern[3][0]]
		v0 = SHAPE_PRIOR[pattern[0][0]][pattern[0][1]] + SHAPE_PRIOR[pattern[1][0]][pattern[1][1]] +\
			 SHAPE_PRIOR[pattern[2][0]][pattern[2][1]] + SHAPE_PRIOR[pattern[3][0]][pattern[3][1]]
		return v0 * 2 + v1 if player == 0 else v1 * 2 + v0


class Cand:
	"""候选动作，包含点和值"""
	def __init__(self, point, value=0):
		self.point = point
		self.value = value

	def __repr__(self):
		return f"{pointXY_V(self.point)}: {self.value}"
		
