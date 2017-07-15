## Neural Chess

A simple chess engine which uses a neural network as its evaluation function. This project also includes a small Tkinter GUI application to play against the engine or against the Stockfish 7 chess engine.  It easily beats me and my friends at Chess, but is not ready for international competition.

This chess engine is a simplified implementation of Matthew Lai's [Giraffe chess engine](https://arxiv.org/abs/1509.01549) without reinforcement learning. My implementation uses a simple negamax search algorithm with alpha-beta pruning.

## The Network

The neural network takes a 373-tuple feature which represents the a chess board state and returns a value between 1 and -1 where 1 indicates the game is won for white, 0 indicates the game is a tie, and -1 indices the game is won for black.  

The exact encoding of the board follows the outline presented in Matthew Lai's paper.  It includes global, piece, and square features about the board state  It is designed to help the neural network latch onto important concepts about chess like positional advantage, mobility, and capturing.  While it is theoretically possible for a network to learn these concepts through training alone, it would dramatically increase training time and as training neural networks is inherently a heuristic algorithm, it may be incredibly unlikely to train a competent engine.  Importantly, though, the network is not taught anything about the rules of chess.

The network architecture is build according to Matthew Lai's design, as shown in this diagram from his paper.
![Network Architecture](images/NetworkArch.png?raw=true)

## Training

The network is trained using the KingBase-Lite chess database of 900,000 games. A random board position from each game is chosen, and a random move is applied to that position to expose the network to more imbalanced positions than is typical of high-level games.  The StockFish chess engine evaluation function is used to label the resulting database of 900,000 board positions.

I originally intended to use the above training as bootstrapping and to continue to train it using temporal difference learning by playing the engine against itself. I wasn't able to implement the a fast enough temporal difference learning algorithm in Tensorflow.

## GUI Application

A small Tkinter-based GUI application is included to play against my engine or against the Stockfish engine. Moves can be made by clicking on the piece you want to move and then the space you want to move it to. The GUI can also be used to watch a game between two chess engines.  (Stockfish easily beats mine)

## Screenshots
![Opening](images/opening.png?raw=true)
![Midgame](images/midgame.png?raw=true)
![Lost](images/lost.png?raw=true)

## Requirements

* [Tensorflow](https://www.tensorflow.org/)
* [Python-chess](https://pypi.python.org/pypi/python-chess)
* [Tkinter](https://wiki.python.org/moin/TkInter)
* [Python Imaging Library with Tkinter module](https://github.com/python-pillow/Pillow)
* [Stockfish 7](https://stockfishchess.org/) (to play against StockFish or to train new networks)

A version of [PyStockFish](https://github.com/iamjarret/pystockfish) (a python wrapper for StockFish) and [Sunfish](https://github.com/thomasahle/sunfish) (a python chess engine)
are included in this repository and are not my own work.
