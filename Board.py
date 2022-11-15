from random import getrandbits, seed
from time import time
from typing import List, Set

from public import *


class Board:
	"""棋盘类，全局只初始化一次"""
	
	def __init__(self):
		self.boardSize = 20
		self.cells = [Chess() for _ in range(1024)]
		self.chessCount = 0
		self.who = PIECE_BLACK
		self.opp = PIECE_WHITE
		self.lastAction = -1
		self.preActions = []
		self._zobristTable = [getrandbits(64) for _ in range(2048)]
		self.zobristKey = getrandbits(64)
		self.hall = set()
		self.shapePoint: List[List[Set[int]]] = [[set() for _ in range(15)] for _ in range(2)]
		self.isTerminated = False
		self.initBoard()
	
	def __str__(self):
		"""打印棋盘，方便命令行测试"""
		res = f"Board sized {self.boardSize}.\n"
		res += "   "
		
		for x in range(self.boardSize):
			res += f"{x:2}|"
		res += "\n"
		
		for x in range(self.boardSize):
			res += f"{x:2}: "
			for y in range(self.boardSize):
				cur_pos = makePoint_V(x, y)
				owner = self.whoseChess(cur_pos)
				if owner == -1:
					piece = VISUAL_PIECE_EMPTY
				else:
					if self.lastAction == cur_pos:
						piece = VISUAL_PIECE_BLACK_CUR if owner == PIECE_BLACK else VISUAL_PIECE_WHITE_CUR
					else:
						piece = VISUAL_PIECE_BLACK if owner == PIECE_BLACK else VISUAL_PIECE_WHITE
				res += piece + "| "
			res += "\n"
		return res
	
	def initBoard(self):
		"""初始化棋盘信息"""
		# 1. 初始化棋点的状态
		for i in range(1024):
			if 4 <= pointX(i) < self.boardSize + 4 and 4 <= pointY(i) < self.boardSize + 4:
				self.cells[i].piece = PIECE_EMPTY
			else:
				self.cells[i].piece = PIECE_OUTSIDE
		# 2. 初始化每个棋点的pattern
		for i in range(1024):
			if self.isOnBoard(i):
				for k in range(4):
					ii = i - MOV[k]
					for p in (8, 4, 2, 1):
						if not self.isOnBoard(ii):
							self.cells[i].pattern[k][0] |= p
							self.cells[i].pattern[k][1] |= p
						ii -= MOV[k]
					ii = i + MOV[k]
					for p in (16, 32, 64, 128):
						if not self.isOnBoard(ii):
							self.cells[i].pattern[k][0] |= p
							self.cells[i].pattern[k][1] |= p
						ii += MOV[k]
		# 3. 初始化每个棋点的shape
		for i in range(1024):
			if self.isOnBoard(i):
				self.cells[i].updateShape()
				self.cells[i].update4()
		
		# 4. 初始化棋盘下一步可能棋型
		for i in range(1024):
			if self.isOnBoard(i):
				cur_cell = self.cells[i]
				self.shapePoint[0][cur_cell.shape4[0]].add(i)
				self.shapePoint[1][cur_cell.shape4[1]].add(i)
	
	def whoseChess(self, position):
		"""判断棋子的主人"""
		piece = self.cells[position].piece
		return -1 if (piece == PIECE_EMPTY or piece == PIECE_OUTSIDE) else piece
	
	def isOnBoard(self, position):
		"""判断点在不在棋盘上"""
		return self.cells[position].piece != PIECE_OUTSIDE
	
	def isEmpty(self, position):
		"""判断点是否为空"""
		return self.cells[position].piece == PIECE_EMPTY
	
	def updateHall(self):
		"""落子后，更新棋盘的‘包’：候选点区域，半径为2"""
		hall = self.hall
		last = self.lastAction
		cells = self.cells
		for offset in NEIGHBOR_2:
			neighborPos = last + offset
			neighborCell = cells[neighborPos]
			neighborCell.neighbor += 1
			if neighborCell.piece == PIECE_EMPTY:
				hall.add(neighborPos)
		if self.chessCount != 1:
			hall.discard(last)
	
	def revertHall(self):
		"""提子后，更新棋盘的'包'"""
		hall = self.hall
		last = self.lastAction
		cells = self.cells
		for offset in NEIGHBOR_2:
			neighborPos = last + offset
			neighborCell = cells[neighborPos]
			neighborCell.neighbor -= 1
			if neighborCell.neighbor == 0 and neighborCell.piece == PIECE_EMPTY:
				hall.discard(neighborPos)
		if self.chessCount != 0:
			hall.add(last)
	
	def makeMove(self, position: int):
		"""棋盘落子"""
		assert self.isEmpty(position)
		
		# 使用局部变量
		who = self.who
		curCell = self.cells[position]
		curShape4 = curCell.shape4
		
		# 更新棋局是否结束的标志
		self.isTerminated = curShape4[who] == A
		
		# 去除该点作为空点在棋盘上记录的棋型信息
		self.shapePoint[0][curShape4[0]].discard(position)
		self.shapePoint[1][curShape4[1]].discard(position)
		
		# 更新zobrist值
		self.zobristKey ^= self._zobristTable[position + (who << 10)]
		
		# 改变当前点的棋子
		curCell.piece = who
		
		# 增加已下棋子数
		self.chessCount += 1

		# 更新历史落子点信息
		self.lastAction = position
		self.preActions.append(position)
		
		# 更新棋型编码信息
		self.updateShapes()
		
		# 交换轮次
		self.who ^= 1
		self.opp ^= 1
		
		# 更新“包”
		self.updateHall()
	
		# assert self.checkShape()
	
	def withdraw(self):
		"""棋盘提子"""
		assert self.cells[self.lastAction].piece == self.opp
		
		# 使用局部变量
		last = self.lastAction
		curCell = self.cells[last]
		curShape4 = curCell.shape4
		
		# 撤回后，棋局一定没有结束
		self.isTerminated = False
		
		# 上一个点重新变成空点，需更新其信息
		curCell.updateShape()
		curCell.update4()
		
		# 增加棋盘的空点棋型信息
		self.shapePoint[0][curShape4[0]].add(last)
		self.shapePoint[1][curShape4[1]].add(last)
		
		# 更新zobrist值
		self.zobristKey ^= self._zobristTable[last + (self.opp << 10)]
		
		# 改变上一个点的棋子
		curCell.piece = PIECE_EMPTY
		
		# 减少已下棋子数
		self.chessCount -= 1
		
		# 更新“包”
		self.revertHall()
		
		# 更新轮次信息
		self.who ^= 1
		self.opp ^= 1
		
		# 更新棋型编码信息
		self.updateShapes()
		
		# 更新历史落子点信息
		self.lastAction = -1 if self.chessCount == 0 else self.preActions[self.chessCount - 1]
		self.preActions.pop()
	
		# assert self.checkShape()
	
	def updateShapes(self):
		"""更新落子/提子点周围点的棋型编码信息"""
		
		# 使用局部变量
		cells = self.cells
		who = self.who
		blackShapePoint = self.shapePoint[0]
		whiteShapePoint = self.shapePoint[1]
		last = self.lastAction
		
		# 四个方向更新
		for k in range(4):
			mask = 1 << 8
			dire = MOV[k]
			for step in (-4, -3, -2, -1, 1, 2, 3, 4):
				# line的八个位置，对应八个掩码(128, 64, 32, 16, 8, 4, 2, 1)
				mask >>= 1
				curPos = last + step * dire
				curCell = cells[curPos]
				curCell.pattern[k][who] ^= mask  # 异或：已下变为未下，未下变为已下
				if curCell.piece == PIECE_EMPTY:
					# 先更新单线，再删除旧棋型，更新棋型，最后增加新棋型
					curCell.update1(k)
					curShape4 = curCell.shape4
					blackShapePoint[curShape4[0]].discard(curPos)
					whiteShapePoint[curShape4[1]].discard(curPos)
					curCell.update4()
					blackShapePoint[curShape4[0]].add(curPos)
					whiteShapePoint[curShape4[1]].add(curPos)
	
	def checkShape(self):
		"""检查当前棋型是否正确（Debug用）"""
		n = [[0 for _ in range(15)] for _ in range(2)]
		for i in range(1024):
			if self.cells[i].piece == PIECE_EMPTY:
				n[0][self.cells[i].shape4[0]] += 1
				n[1][self.cells[i].shape4[1]] += 1
				if i in self.shapePoint[0][self.cells[i].shape4[0]] and i in self.shapePoint[1][
					self.cells[i].shape4[1]]:
					continue
				else:
					return False
		return True
	
	def restart(self):
		while self.chessCount:
			self.withdraw()


def main():
	seed(1234)
	board = Board()
	board.makeMove(makePoint_V(10, 10))
	board.withdraw()
	print([pointXY_V(pos) for pos in board.hall])
	print(board.zobristKey)
	board.makeMove(makePoint_V(0, 0))
	board.makeMove(makePoint_V(1, 0))
	
	board.makeMove(makePoint_V(0, 1))
	board.makeMove(makePoint_V(1, 1))
	board.makeMove(makePoint_V(3, 2))
	board.makeMove(makePoint_V(1, 2))
	board.makeMove(makePoint_V(0, 3))
	board.makeMove(makePoint_V(1, 3))
	board.makeMove(makePoint_V(0, 4))
	print(board.zobristKey)
	board.makeMove(makePoint_V(1, 5))
	
	print(board)
	print([pointXY_V(pos) for pos in board.hall])
	print(len(board.hall))
	print(board.zobristKey)
	board.withdraw()
	print(board)
	print(board.isTerminated)
	print([pointXY_V(pos) for pos in board.hall])
	print(len(board.hall))
	
	print(board.cells[makePoint_V(0, 2)].pattern)
	print(board.cells[makePoint_V(1, 4)].pattern)
	
	print(board.cells[makePoint_V(0, 2)].shape)
	print(board.cells[makePoint_V(0, 2)].shape4)
	
	print(board.cells[makePoint_V(1, 4)].shape)
	print(board.cells[makePoint_V(1, 4)].shape4)
	
	time1 = time()
	for i in range(10000):
		board.makeMove(makePoint_V(10, 10))
		board.withdraw()
	print(time() - time1)
	print(board.cells[makePoint_V(0, 2)].prior(0))
	print(board.cells[makePoint_V(0, 2)].prior(1))
	print(board.cells[makePoint_V(1, 4)].prior(0))
	print(board.cells[makePoint_V(1, 4)].prior(1))


if __name__ == "__main__":
	main()
