import sys
import time
import copy
import string

# 보드 크기 상수
BOARD_SIZE = 19

# 알파-베타 탐색에서 사용할 아주 큰 값들
INFINITY = float('inf')
NEG_INFINITY = float('-inf')

def create_board():
    return [['.' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def print_board(board):
    # 열은 A~S, 행은 1~19로 출력
    header = "   " + " ".join(string.ascii_uppercase[:BOARD_SIZE])
    print(header)
    for idx, row in enumerate(board):
        # 행 번호는 1부터 시작 (자리수 맞추기)
        print(f"{idx+1:2d} " + " ".join(row))
    print()

def copy_board(board):
    return copy.deepcopy(board)

def get_legal_moves(board):
    moves = set()
    n = BOARD_SIZE
    # 만약 보드에 돌이 하나도 없으면 중앙 한 곳만 반환
    found = False
    for r in range(n):
        for c in range(n):
            if board[r][c] != '.':
                found = True
                break
        if found:
            break
    if not found:
        return [(n//2, n//2)]
    # 인접한 칸(거리 1 이내)에 돌이 있는 빈 칸만 후보로 선택
    for r in range(n):
        for c in range(n):
            if board[r][c] == '.':
                for dr in range(-1, 2):  # -1, 0, 1 (거리 1 이내)
                    for dc in range(-1, 2):  # -1, 0, 1 (거리 1 이내)
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            if board[nr][nc] != '.':
                                moves.add((r, c))
                                break
                    else:
                        continue
                    break
    return list(moves)

def check_win(board, player):
    # 가로, 세로, 대각선(양방향)에서 5개 연속이면 승리
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player:
                for dr, dc in directions:
                    count = 1
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == player:
                        count += 1
                        if count >= 5:
                            return True
                        nr += dr
                        nc += dc
    return False

def is_terminal(board):
    # 터미널 상태: 누군가 이겼거나, 보드가 꽉 찼을 때
    if check_win(board, 'b') or check_win(board, 'w'):
        return True
    # 빈 칸이 없으면 무승부 상태
    for row in board:
        if '.' in row:
            return False
    return True

def count_sequence(board, r, c, dr, dc, player):
    """
    (r, c)부터 (dr, dc) 방향으로 player의 돌이 연속된 길이와
    양쪽 끝의 개수를 반환합니다.
    """
    n = BOARD_SIZE
    length = 0
    cur_r, cur_c = r, c
    while 0 <= cur_r < n and 0 <= cur_c < n and board[cur_r][cur_c] == player:
        length += 1
        cur_r += dr
        cur_c += dc

    open_ends = 0
    # 앞쪽(시퀀스 시작 전)
    start_r = r - dr
    start_c = c - dc
    if start_r < 0 or start_r >= n or start_c < 0 or start_c >= n or board[start_r][start_c] == '.':
        open_ends += 1
    # 끝쪽(시퀀스 끝난 후)
    if cur_r < 0 or cur_r >= n or cur_c < 0 or cur_c >= n or board[cur_r][cur_c] == '.':
        open_ends += 1

    return length, open_ends

def pattern_weight(length, open_ends):
    """
    길이와 열린 끝(open_ends)에 따라 가중치를 반환합니다.
    (예시는 기본적인 값이며, 실제 과제에서는 더 정교하게 조정할 수 있습니다.)
    """
    if length >= 5:
        return 100000  # 승리 상태
    if length == 4:
        if open_ends == 2:
            return 10000   # 열린 4: 즉시 승리 위협
        elif open_ends == 1:
            return 1000    # 닫힌 4
    if length == 3:
        if open_ends == 2:
            return 1000    # 열린 3: 매우 강력한 수
        elif open_ends == 1:
            return 100     # 닫힌 3
    if length == 2:
        if open_ends == 2:
            return 100     # 열린 2
        elif open_ends == 1:
            return 10      # 닫힌 2
    if length == 1:
        if open_ends == 2:
            return 10
        elif open_ends == 1:
            return 1
    return 0

def evaluate_player(board, player):
    """
    보드 전체에서 주어진 플레이어의 모든 시퀀스를 평가하여 점수를 합산합니다.
    중복 계산을 피하기 위해, 각 방향에서 시퀀스의 시작점(이전 칸이 player가 아닌 경우)만 평가합니다.
    """
    score = 0
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    n = BOARD_SIZE
    for r in range(n):
        for c in range(n):
            if board[r][c] == player:
                for dr, dc in directions:
                    # 이전 칸이 player이면 이미 계산한 시퀀스이므로 건너뜁니다.
                    prev_r = r - dr
                    prev_c = c - dc
                    if 0 <= prev_r < n and 0 <= prev_c < n and board[prev_r][prev_c] == player:
                        continue
                    length, open_ends = count_sequence(board, r, c, dr, dc, player)
                    score += pattern_weight(length, open_ends)
    return score

def evaluate_board(board, computer, opponent):
    """
    향상된 평가 함수:
    - 컴퓨터와 상대방의 점수를 각각 계산한 후 그 차이를 반환합니다.
    """
    # 승리 여부는 check_win에서 이미 처리되므로 여기서는 시퀀스 점수만 계산합니다.
    comp_score = evaluate_player(board, computer)
    opp_score = evaluate_player(board, opponent)
    return comp_score - opp_score


def alpha_beta(board, depth, alpha, beta, maximizing, start_time, time_limit, computer, opponent):
    # 시간 초과 체크
    if time.time() - start_time > time_limit:
        return evaluate_board(board, computer, opponent)
    if depth == 0 or is_terminal(board):
        return evaluate_board(board, computer, opponent)
    
    moves = get_legal_moves(board)
    if maximizing:
        value = NEG_INFINITY
        for move in moves:
            new_board = copy_board(board)
            new_board[move[0]][move[1]] = computer
            value = max(value, alpha_beta(new_board, depth-1, alpha, beta, False, start_time, time_limit, computer, opponent))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # 가지치기
        return value
    else:
        value = INFINITY
        for move in moves:
            new_board = copy_board(board)
            new_board[move[0]][move[1]] = opponent
            value = min(value, alpha_beta(new_board, depth-1, alpha, beta, True, start_time, time_limit, computer, opponent))
            beta = min(beta, value)
            if alpha >= beta:
                break  # 가지치기
        return value

def iterative_deepening(board, time_limit, computer, opponent):
    best_move = None
    depth = 1
    start_time = time.time()
    
    while True:
        # 시간 제한 체크
        if time.time() - start_time > time_limit:
            break
        
        current_best = None
        current_best_score = NEG_INFINITY
        
        moves = get_legal_moves(board)
        for move in moves:
            new_board = copy_board(board)
            new_board[move[0]][move[1]] = computer
            score = alpha_beta(new_board, depth - 1, NEG_INFINITY, INFINITY, False, start_time, time_limit, computer, opponent)
            if score > current_best_score:
                current_best_score = score
                current_best = move
            # 시간 초과 시 중단
            if time.time() - start_time > time_limit:
                break
        
        # 현재 깊이의 탐색이 끝나면 결과를 업데이트
        if current_best is not None:
            best_move = current_best
        depth += 1
        
        if len(moves) == 1:
            break

        # 시간 제한 체크 후 반복 종료
        if time.time() - start_time > time_limit:
            break
    return best_move

def parse_move(move_str):
    # 예시 입력: "J10", "J,10", "J 10"
    move_str = move_str.replace(',', ' ').strip()
    parts = move_str.split()
    if len(parts) != 2:
        raise ValueError("잘못된 형식입니다. 예: J 10")
    col_str, row_str = parts[0].upper(), parts[1]
    # 열: A=0, B=1, ... S=18
    col = string.ascii_uppercase.index(col_str[0])
    row = int(row_str) - 1
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError("좌표 범위를 벗어났습니다.")
    return (row, col)

def move_to_str(move):
    # (row, col)를 "J10" 형식으로 변환 (열: A~S, 행: 1~19)
    col_letter = string.ascii_uppercase[move[1]]
    row_number = move[0] + 1
    return f"{col_letter}{row_number}"

def main():
    print("오목 게임을 시작합니다.")
    
    # 시간 제한 입력 받기
    while True:
        try:
            time_limit = float(input("컴퓨터의 계산 시간 제한(초)을 입력하세요: "))
            if time_limit <= 0:
                print("시간 제한은 양수여야 합니다.")
                continue
            break
        except ValueError:
            print("시간 제한은 숫자(초)로 입력하세요.")
    
    # 플레이어 색상 입력 받기
    while True:
        player_arg = input("당신의 돌 색상을 선택하세요 (b: 흑돌(선공), w: 백돌(후공)): ").lower()
        if player_arg in ['w', 'b']:
            break
        print("플레이어는 'w' 또는 'b'로 입력하세요.")
    
    # 흑(b)은 항상 선 (첫 수)
    if player_arg == 'b':
        computer = 'w'
        human = 'b'
    else:
        computer = 'b'
        human = 'w'
    
    board = create_board()
    current_player = 'b'  # 항상 흑이 선
    
    print(f"컴퓨터: {computer.upper()}, 인간: {human.upper()}")
    print_board(board)
    
    while not is_terminal(board):
        if current_player == computer:
            print("컴퓨터의 차례입니다...")
            move = iterative_deepening(board, time_limit, computer, human)
            if move is None:
                print("시간 내에 결정하지 못했습니다. 무승부 처리합니다.")
                break
            board[move[0]][move[1]] = computer
            print(f"컴퓨터가 {move_to_str(move)}(으)로 두었습니다.")
        else:
            valid = False
            while not valid:
                try:
                    move_input = input("당신의 차례입니다. 좌표를 입력하세요 (예: J 10): ")
                    move = parse_move(move_input)
                    if board[move[0]][move[1]] != '.':
                        print("해당 위치는 이미 사용 중입니다. 다시 입력하세요.")
                    else:
                        valid = True
                except Exception as e:
                    print(e)
            board[move[0]][move[1]] = human
        print_board(board)
        
        # 승리 검사
        if check_win(board, current_player):
            if current_player == computer:
                print("컴퓨터가 이겼습니다!")
            else:
                print("당신이 이겼습니다!")
            return
        # 턴 전환
        current_player = computer if current_player == human else human
    
    print("게임이 종료되었습니다.")

if __name__ == "__main__":
    main()
