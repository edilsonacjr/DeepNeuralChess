import numpy as np
from neural import NeuralGiraffe
import chess
import NeuralChessEngine.chessengine
import NeuralChessEngine.database

def prepareX(arr):
    if len(arr.shape) == 1:
        dict_train = {'input1': arr[:37].reshape(-1,37),
                      'input2': arr[37:37 + 208].reshape(-1,208),
                      'input3': arr[37 + 208:].reshape(-1,128)}
    else:
        dict_train = {'input1' : arr[:,:37],
                      'input2' : arr[:,37:37+208],
                      'input3' : arr[:,37+208:]}
    return dict_train


def main():
    need_train = False
    if need_train:
        trainX = prepareX(np.load('NeuralChessEngine/gameDB/trainX.npy'))
        trainY = np.load('NeuralChessEngine/gameDB/sfTrainY.npy')
        testX = prepareX(np.load('NeuralChessEngine/gameDB/testX.npy'))
        testY = np.load('NeuralChessEngine/gameDB/sfTestY.npy')

        nn = NeuralGiraffe()
        nn.build()
        nn.fit(trainX, trainY)
        nn.save_model('nn_v0')

    nn = NeuralGiraffe()
    nn.load_model('nn_v0')
    model = nn.get_model()

    board = chess.Board('rnb1k1nr/pppp1ppp/8/2b1p3/4P3/1PN5/PBPP1qPP/R2QKBNR w KQkq - 0 5')
    print(board)
    X = prepareX(NeuralChessEngine.database.getBoardFeature(board))
    print('Current move score is %.2f' % (model.predict(X, batch_size=1)))
    return
    return
    while not board.is_checkmate():
        print(board)
        if board.turn == chess.WHITE:
            mv = input('')
            if chess.Move.from_uci(mv) in board.legal_moves:
                board.push(chess.Move.from_uci(mv))
        else:
            best_move = None
            best_eval = 0
            for move in board.legal_moves:
                new_board = board.copy()
                new_board.push(move)
                X = prepareX(NeuralChessEngine.database.getBoardFeature(new_board))
                print('Current move score is %.2f' % (model.predict(X, batch_size = 1)))
                print(new_board)
                if best_move is None or model.predict(X, batch_size = 1) < best_eval:
                    best_move = move
                    best_eval = model.predict(X, batch_size = 1)
                    print('Best move updated with a score of %.2f' % (best_eval))
            board.push(best_move)
            print('Best move has a score of %.2f' % (best_eval))


if __name__ == '__main__':
    main()