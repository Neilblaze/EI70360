import numpy as np

def is_winner(board, player):
    for i in range(3):
        if all(board[i, j] == player for j in range(3)):
            return True
        if all(board[j, i] == player for j in range(3)):
            return True
    if all(board[i, i] == player for i in range(3)):
        return True
    if all(board[i, 2-i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return not any(board[i, j] == '' for i in range(3) for j in range(3))

def get_available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == '']

def minimax_tictactoe(board: np.ndarray, player: str) -> tuple:
    opponent = 'O' if player == 'X' else 'X'
    
    def minimax(board, depth, is_maximizing):
        if is_winner(board, player):
            return 1
        elif is_winner(board, opponent):
            return -1
        elif is_full(board):
            return 0
        
        if is_maximizing:
            best_score = -float('inf')
            for move in get_available_moves(board):
                i, j = move
                board[i, j] = player
                score = minimax(board, depth + 1, False)
                board[i, j] = ''
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for move in get_available_moves(board):
                i, j = move
                board[i, j] = opponent
                score = minimax(board, depth + 1, True)
                board[i, j] = ''
                best_score = min(score, best_score)
            return best_score
    
    best_score = -float('inf')
    best_move = None
    
    for move in get_available_moves(board):
        i, j = move
        board[i, j] = player
        score = minimax(board, 0, False)
        board[i, j] = ''
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move
