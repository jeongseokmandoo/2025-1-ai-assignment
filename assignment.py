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
    # 인접한 칸(거리 2 이내)에 돌이 있는 빈 칸만 후보로 선택
    for r in range(n):
        for c in range(n):
            if board[r][c] == '.':
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
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

def evaluate_board(board, computer, opponent):
    # 단순 평가 함수: 만약 승리면 매우 큰 값 반환, 아니면 돌의 개수 차이 계산
    if check_win(board, computer):
        return 100000
    if check_win(board, opponent):
        return -100000
    comp_count = sum(row.count(computer) for row in board)
    opp_count = sum(row.count(opponent) for row in board)
    return comp_count - opp_count

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
    best_score = NEG_INFINITY
    depth = 5 # 최대 깊이 제한
    start_time = time.time()
    
    while True:
        # 시간 제한 체크
        if time.time() - start_time > time_limit:
            break
        
        current_best = None
        current_best_score = NEG_INFINITY
        
        moves = get_legal_moves(board)
        # 간단한 무브 오더링: 여기서는 그냥 legal moves 리스트 순서 그대로 탐색 (추후 개선 가능)
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
            best_score = current_best_score
        depth += 1
        
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
