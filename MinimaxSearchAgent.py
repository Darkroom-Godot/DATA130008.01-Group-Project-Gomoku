from random import randint, seed
from time import time
from typing import List

from Board import Board
from public import *


class MinimaxSearchAgent:
	"""AI类，需要绑定一个棋盘Board类"""
	
	def __init__(self, board: Board, max_depth=MAX_DEPTH):
		self.board = board
		self.transTable = {}
		self.max_depth = max_depth
		self.rootBest: Cand = Cand(-1)
		self.rootCand: List[Cand] = []
		self.timeout_turn = TIME_TURN
		self.time_left = TIME_MATCH
		self.startTime = 0
		self.stopThinking = False
		self.cnt = 500
		self.t_VCT_Start = 0
		self.hit = 0

	def getTime(self):
		"""此步已经思考的时间"""
		return time() - self.startTime

	def getStopTime(self):
		"""每步最大思考时间"""
		return min(self.timeout_turn, self.time_left / 7)

	def recordTransTable(self, depth, value, flag, best=-1):
		"""将搜索完毕的深度、评估值、哈希标志、最优动作记录到置换表"""
		if (item := self.transTable.get(self.board.zobristKey, None)) is None or item[0] <= depth:
			self.transTable[self.board.zobristKey] = (depth, value, flag, best)

	def recordKillingAction(self, best):
		self.transTable[self.board.zobristKey] = (KILL_DEPTH, WIN_SCORE, HASH_EXACT, best)

	def probeTransTable(self, depth, alpha, beta):
		"""探查置换表，若发现深度更深的，则返回评估值"""
		if beta == alpha + 1:
			return HASH_UNKNOWN
		if (item := self.transTable.get(self.board.zobristKey, None)) is not None:
			itemDepth, itemValue, itemFlag, itemBest = item
			if itemFlag == HASH_EXACT and (itemValue == WIN_SCORE or itemValue == LOSE_SCORE):
				return itemValue
			if itemDepth >= depth:
				if itemFlag == HASH_EXACT:
					return itemValue
				elif itemFlag == HASH_ALPHA and itemValue <= alpha:
					return alpha
				elif itemFlag == HASH_BETA and itemValue >= beta:
					return beta
		return HASH_UNKNOWN

	def find_killingAction(self):
		"""已经算杀成功时，直接从置换表中取出最佳动作"""
		if (item := self.transTable.get(self.board.zobristKey, None)) is not None:
			if item[0] == KILL_DEPTH and item[1] == WIN_SCORE and item[2] == HASH_EXACT:
				self.rootBest.point = item[3]
				return True
		return None

	def evaluate(self):
		"""深度到达上限，评估棋盘，返回得分"""
		board = self.board
		who = board.who
		opp = board.opp
		shapePointWho = board.shapePoint[who]
		shapePointOpp = board.shapePoint[opp]
		
		if shapePointWho[A]:
			return WIN_SCORE
		if (nOppA := len(shapePointOpp[A])) >= 2:
			return LOSE_SCORE
		if shapePointWho[B] and nOppA == 0:
			return WIN_SCORE
		
		if (not shapePointOpp[B]) and (not shapePointOpp[C]) and (not shapePointOpp[D]) and (not shapePointOpp[E]) and (not shapePointOpp[F]):
			if shapePointWho[C] or shapePointWho[G]:
				return WIN_SCORE
		
		scores = [0, 0]
		for pre_action in board.preActions:
			cur_cell = board.cells[pre_action]
			cur_piece = cur_cell.piece
			cur_pattern = cur_cell.pattern
			for k in range(4):
				scores[cur_piece] += SHAPE_RANK[cur_pattern[k][cur_piece]][cur_pattern[k][1 - cur_piece]]
		return scores[board.who] - scores[board.opp] + 50
	
	def generateCand(self, candList):
		"""生成当前棋盘的候选动作：精确缩减、置换表提案、防守缩减"""
		board = self.board
		who = board.who
		opp = board.opp
		shapePointWho = board.shapePoint[who]
		shapePointOpp = board.shapePoint[opp]
		setAWho = shapePointWho[A]
		setAOpp = shapePointOpp[A]
		setBWho = shapePointWho[B]
		setBOpp = shapePointOpp[B]

		# 精确缩减：我方连五、对方连五、我方连四，无需考虑其他
		if shapePointWho[A]:
			setAWho.add(point := setAWho.pop())
			candList.append(Cand(point))
			return
		
		if shapePointOpp[A]:
			setAOpp.add(point := setAOpp.pop())
			candList.append(Cand(point))
			return
		
		if shapePointWho[B]:
			setBWho.add(point := setBWho.pop())
			candList.append(Cand(point))
			return

		# 置换表提案：找出历史记录的最佳点，赋予最高分值以将其排在首位
		hashMove = -1
		if (item := self.transTable.get(board.zobristKey, None)) is not None and (hashMove := item[3]) != -1:
			candList.append(Cand(point=hashMove, value=WIN_SCORE))

		# 防守缩减：若对方已经活三，将我方冲四点、对方活四冲四点列为可选
		if (oppF4Num := len(setBOpp)) > 0:
			
			# 对手活四（必须防守）
			candList.extend([Cand(point=i, value=board.cells[i].prior(who)) for i in setBOpp if i != hashMove])
			
			# 我方冲四（可以延缓失败，或进行算杀）
			candList.extend([
				Cand(point=i, value=board.cells[i].prior(who))
				for i in shapePointWho[C] | shapePointWho[D] | shapePointWho[E] | shapePointWho[F]
				if i != hashMove
			])
			
			if oppF4Num == 1:  # _XX_X_型活三，只有一个活四和两个冲四可以作为防守点
				# 对手冲四（距离为{2, 3}是为了识别当前棋型的两个冲四点，而不考虑其他眠三棋型形成的冲四）
				setBOpp.add(BPointOpp := setBOpp.pop())
				candList.extend([
					Cand(point=i, value=board.cells[i].prior(who))
					for i in shapePointOpp[C] | shapePointOpp[D] | shapePointOpp[E] | shapePointOpp[F]
					if i != hashMove and ((dist := distance(i, BPointOpp)) == 2 or dist == 3)
				])
		else:
			# 无缩减，添加所有候选点
			candList.extend([Cand(point=i, value=board.cells[i].prior(who)) for i in board.hall if i != hashMove])

		candList.sort(key=lambda x: x.value, reverse=True)
		for _ in range(len(candList)-1, MAX_ACTION_NUM-1, -1):
			candList.pop()

	def delLoseCand(self):
		"""删除根节点的必败动作，后续迭代加深不再考虑"""
		candList = self.rootCand
		n = len(candList)
		for i in range(n - 1, -1, -1):
			if candList[i].value <= LOSE_SCORE:
				for j in range(i + 1, n):
					candList[j - 1] = candList[j]
				n -= 1
				candList.pop()
	
	def delVctLose(self):
		"""删除根节点候选动作中会被对手VCT胜的动作"""
		maxLoseSteps = 0
		bestPoint = self.rootCand[0].point
		isAllLose = True
		
		for cur_cand in self.rootCand:
			self.board.makeMove(cur_cand.point)  # 先走这一步
			if (result := self.vctKiller()) > 0:
				cur_cand.value = LOSE_SCORE  # 该步会被对手VCT获胜
				if isAllLose and result > maxLoseSteps:  # 如果都是必败，选择拖延步数最长的为最佳点
					maxLoseSteps = result
					bestPoint = cur_cand.point
			else:  # 该步不会被VCT获胜
				if isAllLose:
					isAllLose = False
					bestPoint = cur_cand.point
			self.board.withdraw()
		self.delLoseCand()  # 删除会失败的结点
		return bestPoint
	
	def alphaBetaSearch(self, depth, alpha, beta):
		"""非根节点的 Alpha-Beta 搜索"""
		
		# 时间控制
		self.cnt -= 1
		if self.cnt == 0:
			self.cnt = 500
			if self.getTime() + 0.2 > self.getStopTime():  # 接近时间上限时，标记停止搜索，返回当前最佳值
				self.stopThinking = True
				return alpha
		
		board = self.board
		
		# 若对手上一步棋已经赢棋，返回失败分，不再进行搜索
		if board.isTerminated:
			return LOSE_SCORE

		# 第一步：探查置换表。若探查到，直接返回表中的值
		if (hashValue := self.probeTransTable(depth, alpha, beta)) != HASH_UNKNOWN:
			self.hit += 1
			return hashValue
		
		q = self.quickKiller()
		if q != 0:
			return WIN_SCORE if q > 0 else LOSE_SCORE
		
		if depth <= 0:  # 深度到达上限，评估棋盘，记录到置换表
			EvalValue = self.evaluate()
			self.recordTransTable(depth, EvalValue, HASH_EXACT)
			return EvalValue

		# 6. 递归搜索
		hash_flag = HASH_ALPHA  # 置换表标志
		foundPV = False  # PVS标志：是否找到PV结点
		candList = []
		self.generateCand(candList)
		best = candList[0]
		for cur_cand in candList:
			board.makeMove(cur_cand.point)
			if foundPV and alpha + 1 < beta:  # 如果找到PV结点，则用(alpha-1, alpha)的窗口进行Alpha-Beta搜索，提升剪枝效率
				value = -self.alphaBetaSearch(depth - 1, -alpha - 1, -alpha)
				if alpha < value < beta:  # 发现一个同样是PV结点的结点，分数不准确，重新用(alpha, beta)窗口搜索
					value = -self.alphaBetaSearch(depth - 1, -beta, -alpha)
			else:  # 没有找到PV结点，正常搜索
				value = -self.alphaBetaSearch(depth - 1, -beta, -alpha)
			board.withdraw()
			if value >= beta:
				if not self.stopThinking and beta != alpha + 1:  # 若停止搜索，或在PVS搜索过程中，该结果可能是不准确的，因此不予记录
					self.recordTransTable(depth, beta, HASH_BETA, cur_cand.point)
				return beta
			if value > alpha:  # 在(alpha, beta)窗口内，是PV结点
				foundPV = True  # 改变PVS标志，下一步使用PVS搜索
				hash_flag = HASH_EXACT  # 更新置换表标志、当前最佳候选动作和alpha值
				best = cur_cand
				alpha = value
			if self.stopThinking:
				break

		if not self.stopThinking and beta != alpha + 1:  # 搜索完毕，将值和最佳动作添加到置换表
			self.recordTransTable(depth, alpha, hash_flag, best.point)

		return alpha

	def rootSearch(self, depth, alpha, beta):
		"""根结点的搜索（相比非根结点，额外记录控制迭代加深的信息）"""

		if len(self.rootCand) == 1:  # 只有一个候选动作，直接返回
			self.stopThinking = True
			return self.rootCand[0]

		bestCand = Cand(point=-1, value=alpha - 1)
		foundPV = False
		for curCand in self.rootCand:
			self.board.makeMove(curCand.point)
			if foundPV and alpha + 1 < beta:
				value = -self.alphaBetaSearch(depth - 1, -alpha - 1, -alpha)
				if alpha < value < beta:
					value = -self.alphaBetaSearch(depth - 1, -beta, -alpha)
			else:
				value = -self.alphaBetaSearch(depth - 1, -beta, -alpha)
			self.board.withdraw()
			if self.stopThinking:
				break
			curCand.value = value
			if value > bestCand.value:
				foundPV = True
				bestCand = curCand
				alpha = value
				if value >= WIN_SCORE:
					self.stopThinking = True
					break
		return bestCand
	
	def quickKiller(self):
		"""已有威胁，尝试进行快速算杀"""
		board = self.board
		who = board.who
		opp = board.opp
		shapePointWho = board.shapePoint[who]
		shapePointOpp = board.shapePoint[opp]
		setAWho = shapePointWho[A]
		setAOpp = shapePointOpp[A]
		setBWho = shapePointWho[B]
		setBOpp = shapePointOpp[B]
		setCWho = shapePointWho[C]
		setGWho = shapePointWho[G]
		
		if setAWho:  # 我方有连五点，1步获胜
			setAWho.add(winPoint := setAWho.pop())
			self.recordKillingAction(winPoint)
			return 1
		if (nAOpp := len(setAOpp)) >= 2:  # 对方有两个以上连五点，2步失败
			return -2
		if nAOpp == 1:  # 对方有一个连五点，尝试防守
			setAOpp.add(point := setAOpp.pop())
			board.makeMove(point)
			q = -self.quickKiller()  # 交换攻防递归搜索
			board.withdraw()
			if q < 0:
				return q - 1
			elif q > 0:  # 有一个点可以反败为胜，则获胜
				self.recordKillingAction(point)
				return q + 1
			else:
				return 0
		if setBWho:  # 我方有活四点，3步获胜
			setBWho.add(winPoint := setBWho.pop())
			self.recordKillingAction(winPoint)
			return 3
		
		# 为了快速返回，将必杀情况判断移到前面来
		if (not setBOpp) and (not shapePointOpp[C]) and (not shapePointOpp[D]):
			if setCWho:  # 对方没有连冲四攻击机会，我方有冲四活三点，5步获胜
				setCWho.add(winPoint := setCWho.pop())
				self.recordKillingAction(winPoint)
				return 5
			if setGWho:  # 我方有双活三点，5步获胜
				setGWho.add(winPoint := setGWho.pop())
				self.recordKillingAction(winPoint)
				return 5
		if setCWho:  # 尝试我方冲四活三点
			for point in setCWho:
				board.makeMove(point)
				q = -self.quickKiller()
				board.withdraw()
				if q > 0:
					self.recordKillingAction(point)
					return q + 1
		
		return 0  # 没有找到杀棋，返回0
	
	def vcfKiller(self):
		"""尝试进行VCF算杀"""
		board = self.board
		if board.chessCount < 10:  # 棋子数少，不进行算杀
			return 0
		who = board.who
		shapePointWho = board.shapePoint[who]
		# 判断是否有可能冲四或活四，若无，则无法进行VCF算杀
		if shapePointWho[A] or shapePointWho[B] or shapePointWho[C] or shapePointWho[D] or shapePointWho[E] or shapePointWho[F]:
			return self.vcfSearch(who, MAX_VCF_DEPTH)
		return 0
	
	def vctKiller(self, maxDepth=MAX_VCT_DEPTH):
		"""尝试进行VCT算杀：最复杂的算杀"""
		if self.time_left < 8:  # 当前局时不足，算杀难以控制时间，放弃算杀
			return 0
		board = self.board
		if board.chessCount < 10:
			return 0
		who = board.who
		shapePointWho = board.shapePoint[who]
		if shapePointWho[A] or shapePointWho[B] or shapePointWho[C] or shapePointWho[D] or shapePointWho[E] or shapePointWho[F] \
			or shapePointWho[G] or shapePointWho[H] or shapePointWho[I] or shapePointWho[J]:
			self.t_VCT_Start = time()
			for depth in range(8, maxDepth + 1, 2):  # 迭代加深地进行VCT算杀，因为VCT可能导致算杀爆炸
				result = self.vctSearch(who, depth)
				if result > 0 or (time() - self.t_VCT_Start) * 5 >= MAX_VCT_TIME:  # 控制时间
					return result
		return 0
	
	def vcfSearch(self, searcher, depth):
		"""VCF算杀主函数"""
		if depth <= 0:
			return 0
		
		board = self.board
		who = board.who
		opp = board.opp
		shapePointWho = board.shapePoint[who]
		shapePointOpp = board.shapePoint[opp]
		setAWho = shapePointWho[A]
		setAOpp = shapePointOpp[A]
		setBWho = shapePointWho[B]
		setBOpp = shapePointOpp[B]
		setCWho = shapePointWho[C]
		setDWho = shapePointWho[D]
		setEWho = shapePointWho[E]
		setGWho = shapePointWho[G]
		
		if setAWho:
			setAWho.add(winPoint := setAWho.pop())
			self.recordKillingAction(winPoint)
			return 1
		if (nAOpp := len(setAOpp)) >= 2:
			return -2
		if nAOpp == 1:
			setAOpp.add(point := setAOpp.pop())
			board.makeMove(point)
			q = -self.vcfSearch(searcher, depth - 1)
			board.withdraw()
			if q < 0:
				return q - 1
			elif q > 0:
				self.recordKillingAction(point)
				return q + 1
			else:
				return 0
		if setBWho:
			setBWho.add(winPoint := setBWho.pop())
			self.recordKillingAction(winPoint)
			return 3
		
		if who == searcher:
			# 为了快速返回，将必杀情况判断移到前面来
			if (not setBOpp) and (not shapePointOpp[C]) and (not shapePointOpp[D]) and (not shapePointOpp[E]) and (not shapePointOpp[F]):
				if setCWho:
					setCWho.add(winPoint := setCWho.pop())
					self.recordKillingAction(winPoint)
					return 5
				if setGWho:
					setGWho.add(winPoint := setGWho.pop())
					self.recordKillingAction(winPoint)
					return 5
			
			if setCWho:
				for point in setCWho:
					board.makeMove(point)
					q = -self.vcfSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
			
			if setDWho:
				for point in setDWho:
					board.makeMove(point)
					q = -self.vcfSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
			
			if setEWho:
				for point in setEWho:
					board.makeMove(point)
					q = -self.vcfSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
			
		return 0
	
	def vctSearch(self, searcher, depth):
		"""VCT算杀主函数"""
		if depth <= 0:
			return 0
		
		board = self.board
		who = board.who
		opp = board.opp
		shapePointWho = board.shapePoint[who]
		shapePointOpp = board.shapePoint[opp]
		setAWho = shapePointWho[A]
		setAOpp = shapePointOpp[A]
		setBWho = shapePointWho[B]
		setBOpp = shapePointOpp[B]
		setCWho = shapePointWho[C]
		setDWho = shapePointWho[D]
		setGWho = shapePointWho[G]
		setHWho = shapePointWho[H]
		setIWho = shapePointWho[I]
		
		if setAWho:
			setAWho.add(winPoint := setAWho.pop())
			self.recordKillingAction(winPoint)
			return 1
		if (nAOpp := len(setAOpp)) >= 2:
			return -2
		if nAOpp == 1:
			setAOpp.add(point := setAOpp.pop())
			board.makeMove(point)
			q = -self.vctSearch(searcher, depth - 1)
			board.withdraw()
			if q < 0:
				return q - 1
			elif q > 0:
				self.recordKillingAction(point)
				return q + 1
			else:
				return 0
		if setBWho:
			setBWho.add(winPoint := setBWho.pop())
			self.recordKillingAction(winPoint)
			return 3
		
		if who == searcher:
			# 为了快速返回，将必杀情况判断移到前面来
			if (not setBOpp) and (not shapePointOpp[C]) and (not shapePointOpp[D]) and (not shapePointOpp[E]) and (not shapePointOpp[F]):
				if setCWho:
					setCWho.add(winPoint := setCWho.pop())
					self.recordKillingAction(winPoint)
					return 5
				if setGWho:
					setGWho.add(winPoint := setGWho.pop())
					self.recordKillingAction(winPoint)
					return 5
			
			if setCWho:
				for point in setCWho:
					board.makeMove(point)
					q = -self.vctSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
			
			if setDWho:
				for point in setDWho:
					board.makeMove(point)
					q = -self.vctSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
					
			if setGWho:
				for point in setGWho:
					board.makeMove(point)
					q = -self.vctSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
			
			if setHWho:
				for point in setHWho:
					board.makeMove(point)
					q = -self.vctSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
			
			if setIWho:
				for point in setIWho:
					board.makeMove(point)
					q = -self.vctSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
			
		else:
			if setBOpp:
				max_q = -100
				pointList = list(setBOpp | setCWho | setDWho | shapePointWho[E] | shapePointWho[F])
				if len(setBOpp) == 1:
					# 对手冲四
					setBOpp.add(BPointOpp := setBOpp.pop())
					pointList.extend([
						i for i in shapePointOpp[C] | shapePointOpp[D] | shapePointOpp[E] | shapePointOpp[F]
						if (dist := distance(i, BPointOpp)) == 2 or dist == 3
					])
				for point in pointList:
					board.makeMove(point)
					q = -self.vctSearch(searcher, depth - 1)
					board.withdraw()
					if q > 0:
						self.recordKillingAction(point)
						return q + 1
					elif q == 0:
						return 0
					elif q > max_q:
						max_q = q
				return max_q
		
		return 0
	
	def solve(self):
		"""主要搜索函数：从棋盘返回动作"""
		self.startTime = time()
		
		# 第一步下中间，第二步随机下旁边
		if self.board.chessCount == 0:
			self.rootBest.point = makePoint_V(10, 10)
		elif self.board.chessCount == 1:
			first_x, first_y = pointXY(self.board.lastAction)
			while True:
				rand_dx = randint(-1, 1)
				rand_dy = randint(-1, 1)
				if rand_dx != 0 or rand_dy != 0:
					break
			self.rootBest.point = makePoint(first_x + rand_dx, first_y + rand_dy)

		else:
			# 尝试查找是否已经算杀成功，若找到，则直接返回动作
			solved = self.find_killingAction()
			
			# 先进行VCF算杀
			if not solved and self.vcfKiller() > 0:
				self.find_killingAction()
				solved = True
			
			# 再进行VCT算杀
			if not solved and self.vctKiller() > 0:
				self.find_killingAction()
				solved = True
			
			# 算杀攻击不成功，生成根节点候选动作
			if not solved:
				rootCand = self.rootCand
				rootCand.clear()
				self.generateCand(rootCand)
				bestPoint = self.delVctLose()  # 防守敌方的VCT，去掉失败的候选动作
				if len(rootCand) <= 1:  # 若无法防守，选择拖延步数最长的动作
					solved = True
					self.rootBest.point = bestPoint
			
			# 进入极小极大搜索截断
			if not solved:
				self.stopThinking = False
				for depth in range(2, self.max_depth + 1, 1):  # 迭代加深循环
					best = self.rootSearch(depth, LOSE_SCORE, WIN_SCORE)  # 对每个深度进行根节点搜索
					if best.point != -1:
						self.rootBest = best
					usedTime = self.getTime()  # 控制时间
					if self.stopThinking or self.rootBest.value == LOSE_SCORE or usedTime * 5 > self.getStopTime():
						break
					else:  # 删除失败动作，并按本轮结果排序，以提高下一次加深的搜索的剪枝效率
						self.delLoseCand()
						self.rootCand.sort(key=lambda cand: cand.value, reverse=True)
		
		# 搜索完毕，更新剩余局时
		self.time_left -= self.getTime()
		return self.rootBest.point


def main():
	seed(1234)
	board = Board()
	black_agent = MinimaxSearchAgent(board, max_depth=8)
	white_agent = MinimaxSearchAgent(board, max_depth=8)
	
	time1 = time()
	while True:
		action = black_agent.solve()
		board.makeMove(action)
		print(board)
		if board.isTerminated:
			break
		if action == makePoint_V(7, 11):
			print()
		
		action = white_agent.solve()
		board.makeMove(action)
		print(board)
		if board.isTerminated:
			break
		if action == makePoint_V(10, 12):
			print()
	
	print(f"Total time: {time() - time1}")
	print("Winner: {}".format("Black" if board.opp == PIECE_BLACK else "White"))
	print(black_agent.hit)
	print(white_agent.hit)


if __name__ == "__main__":
	main()
