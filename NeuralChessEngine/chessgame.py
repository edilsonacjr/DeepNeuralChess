# chessgame.py
#
# Simple chess game using Tkinter and python-chess
# Written by Kyle McDonell
#
# CS 251
# Spring 2016


import Tkinter as tk
import PIL
from PIL import Image, ImageTk
import chess
import chessengine
import neuralnetwork
import threading
import os


# Function to clear the console
clear = lambda: os.system('cls' if os.name == 'nt' else 'clear')

# The visual board
class TkChessBoard(tk.Frame):

    # Create the board with the given AI
    def __init__(self, parent, board, whiteAI=None, blackAI=None):
        tk.Frame.__init__(self, parent)

        # Create the board and the canvas
        self.board = board
        self.squareSize = 64
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=self.squareSize * 8, height=self.squareSize * 8)
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)

        # This binding will cause a refresh if the user interactively
        # changes the window size
        self.canvas.bind("<Configure>", self.refresh)

        # Load the piece pictures:
        self.piecePictures = {}
        self.tempPictures = []
        pieces = ['b','k','n','p','q','r']
        for piece in pieces:
            self.piecePictures[piece.upper()] = Image.open('pieces/white' + piece + '.png')
            self.piecePictures[piece.lower()] = Image.open('pieces/black' + piece + '.png')

        # User click binding
        self.clickSquare = None
        self.canvas.bind("<Button-1>", self.mouseClick)

        # AI dict
        self.AI = {True: whiteAI, False: blackAI}

        # If there is a white AI, have him start his turn
        if whiteAI is not None:
            thread = threading.Thread(target=whiteAI.makeMove,
                                      args=(self.board, self.makeMove))
            thread.daemon = True
            thread.start()


    # Draw the underlying board squares
    def drawBoard(self):
        self.canvas.delete("square")
        for row in xrange(8):
            for col in xrange(8):
                color = 'white' if (row+col)%2 == 0 else 'gray'
                x1 = (col * self.squareSize)
                y1 = (row * self.squareSize)
                x2 = x1 + self.squareSize
                y2 = y1 + self.squareSize
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black",
                                             fill=color, tags="square")

    # Draw the pieces on the board
    def drawPieces(self):
        self.canvas.delete("piece")
        self.tempPictures = []
        fen = self.board.fen().split(' ')[0]
        for rowIndex, row in enumerate(fen.split('/')):
            # Index on the board and index in the row string
            colIndex = 0
            index = 0
            # For each square on the board
            while colIndex < 8:
                piece = row[index]
                # If the char is a number, skip that many column indices
                if piece.isdigit():
                    colIndex += int(piece)
                # Otherwise, draw the piece at rowIndex, colIndex
                else:
                    x = colIndex * self.squareSize + self.squareSize/2
                    y = rowIndex * self.squareSize + self.squareSize/2
                    # Resize the piece picture
                    image = self.piecePictures[piece]
                    wpercent = (self.squareSize / float(image.size[0]))
                    hsize = int((float(image.size[1]) * float(wpercent)))
                    image = image.resize((self.squareSize, hsize), PIL.Image.ANTIALIAS)
                    self.tempPictures.append(ImageTk.PhotoImage(image))

                    # Draw the piece
                    self.canvas.create_image(x, y, image=self.tempPictures[-1], tags="piece")

                    colIndex += 1
                index += 1


    # Redraw the board, including when the window is resized
    def refresh(self, event=None):
        if event is not None:
            self.squareSize = min(int((event.width - 1) / 8), int((event.height - 1) / 8))
        self.drawBoard()
        self.drawPieces()

    # Allow the user to move via mouse click
    def mouseClick(self, event):
        # If the game is over, return
        if self.board.is_game_over():
            return

        # If there is an AI and they are thinking, reset the click history and return
        if self.AI[self.board.turn] is not None:
            print 'The', ('white' if self.board.turn else 'black'), 'AI is thinking...'
            self.clickSquare = None
            return

        squareInfo = self.getSquare(event.x, event.y)

        # Reset the click history and return if no square was clicked
        if squareInfo is None:
            self.clickSquare = None
            return

        # Or if this is the first click and not the current player's piece
        if self.clickSquare is None:
            if squareInfo[1] is None:
                self.clickSquare = None
                return
            elif squareInfo[1].color != self.board.turn:
                color = 'white' if self.board.turn else 'black'
                print 'It is ' + color + "'s turn."
                self.clickSquare = None
                return

        # Save the click if it is the first one or if it is on the current players piece
        if self.clickSquare is None or \
                (squareInfo[1] is not None and squareInfo[1].color == self.board.turn):
            self.clickSquare = squareInfo[0]
            return

        # Otherwise, try to move the piece
        move = chess.Move.from_uci(self.clickSquare + squareInfo[0])
        self.makeMove(move)
        self.clickSquare = None

    # Get square at the given coordinates
    def getSquare(self, x, y):
        row = y // self.squareSize
        col = x // self.squareSize
        # If is not a square, return None
        if row >= 8 or col >= 8:
            return None
        squarestr = chr(col + ord('a')) + str(8 - row)
        squareid = chess.Move.from_uci(squarestr + 'a1').from_square
        piece = self.board.piece_at(squareid)
        # Return the square string and the piece on the square, or None if there is none
        return squarestr, piece

    # Make a move
    def makeMove(self, move, uci=False):
        if self.board.is_legal(move):
            print self.board.san(move)
            self.board.push(move)
            self.refresh()
            clear()
            self.printMoves()
        # Allow promotions
        elif self.board.is_legal(chess.Move.from_uci(move.uci()+'q')):
            self.board.push(chess.Move.from_uci(move.uci() + 'q'))
            self.refresh()
            clear()
            self.printMoves()
        else:
            print move.uci() + ' is not a valid move.'

        if self.board.is_game_over():
            clear()
            print "Game Over: " + self.board.result()
            self.printMoves()
            return

        # If there is an AI, have it make its move
        AI = self.AI[self.board.turn]
        if AI is not None:
            thread = threading.Thread(
                target=AI.makeMove, args=(chess.Board(self.board.fen()), self.makeMove))
            thread.daemon = True
            thread.start()

    # Print the current list of moves.  The python_chess method cannot be used because
    # it breaks with castling!
    def printMoves(self, event=None):
        string = ''
        moveList = []
        while len(self.board.move_stack) > 0:
            mv = self.board.pop()
            moveList.append(mv)
            string = board.san(mv) + ' ' + string
            if len(self.board.move_stack) % 2 == 0:
                string = str(len(self.board.move_stack)/2+1) + '. ' + string
        print string

        for mv in reversed(moveList):
            board.push(mv)




if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chess")
    board = chess.Board()
    #gameboard = TkChessBoard(root, board, whiteAI=neuralnetwork.NeuralNet(), blackAI=chessengine.BabyStockFish())
    gameboard = TkChessBoard(root, board, blackAI=neuralnetwork.NeuralNet())
    gameboard.pack(side="top", fill="both", expand="true", padx=4, pady=4)
    root.mainloop()
