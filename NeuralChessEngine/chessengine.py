# chessengine.py
#
# Chessengine class using alphabeta minimax searching
# Written by Kyle McDonell
#
# CS 251
# Spring 2016

from __future__ import print_function
import pystockfish
import chess
import numpy as np


# An abstract chess engine class
class ChessEngine(object):

    # The depth to be used for minimax searching
    def __init__(self, depth=1):
        self.depth = depth

    # Make a move given the current board, returning the move or calling the
    # provided function
    def makeMove(self, board, moveFunction=None):
        move = self.alphabeta(
            board, self.depth, float('-infinity'), float('infinity'))[1]
        if moveFunction is not None:
            moveFunction(move)
        else:
            return move

    # Search for the best move given the current board. Returns a tuple, (score, bestMove)
    # Alpha-beta pruning elimintates subtrees which cannot contain a better move than
    # the best currently found.  Code adapted from stackoverflow
    def alphabeta(self, board, depth, alpha, beta):

        # If the game is over, return the score and no move
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1, None
            elif result == "0-1":
                return -1, None
            elif result == "1/2-1/2":
                return 0, None
            else:
                print("Unknown board result!")
                return 0, None

        # If no depth should be searched, don't return a move, just evaluate the board
        if depth == 0: return self.evaluate(board), None

        # If it is whites turn
        if board.turn:
            bestmove = None
            for move in board.legal_moves:
                board.push(move)
                score, submove = self.alphabeta(board, depth - 1, alpha, beta)
                board.pop()
                # White maximizes score
                if score > alpha:
                    alpha = score
                    bestmove = move
                    # alpha-beta cutoff
                    if alpha >= beta:
                        break
            return alpha, bestmove
        else:
            bestmove = None
            for move in board.legal_moves:
                board.push(move)
                score, submove = self.alphabeta(board, depth - 1, alpha, beta)
                board.pop()
                # Black minimizes his score
                if score < beta:
                    beta = score
                    bestmove = move
                    # Alpha-beta cutoff
                    if alpha >= beta:
                        break
            return beta, bestmove

    # Subclasses must override the evaluate function
    def evaluate(self, board):
        return 0


# Stockfish chess engine
class BabyStockFish(ChessEngine):

    def __init__(self, depth=15):
        self.engine = pystockfish.Engine(depth=depth)
        ChessEngine.__init__(self, depth)

    # Have stockfish find the best move
    def makeMove(self, board, moveFunction=None):
        self.engine.setfenposition(board.fen())
        move = self.engine.bestmove()["move"]
        if moveFunction is not None:
            moveFunction(chess.Move.from_uci(move))
        else:
            return move

    # Evaluate the board using stockfish's score.  It is converted from centipawns or
    # moves to mate
    def evaluate(self, board):
        self.engine.setfenposition(board.fen())
        info = self.engine.bestmove()["info"]
        endOfScoreIndex = info.find(" nodes")
        endOfScoreIndex = len(info) if endOfScoreIndex == -1 else endOfScoreIndex

        # If the score is in centipawns, get it and transform it to -1 to 1
        if "score cp" in info:
            centipawns = info[info.find("score cp ") + len("score cp "): endOfScoreIndex]
            return np.tanh(int(centipawns) / 833.3) * (int(board.turn) * 2 - 1)

        # If the score is in moves to mate, return the value of the winning side
        elif "score mate" in info:
            mate = info[info.find("score mate ") + len("score mate "): endOfScoreIndex]
            return 1 if mate > 0 else -1

        # Otherwise, there is some format I didn't account for!
        else:
            print("BIG PROBLEM!  We can't find the stockfish score")




