import sys
import time
import copy
import string
import hashlib

# 보드 크기 상수
BOARD_SIZE = 19

# 알파-베타 탐색에서 사용할 아주 큰 값들
INFINITY = float('inf')
NEG_INFINITY = float('-inf')

# 트랜스포지션 테이블 (중복 계산 방지)
transposition_table = {}

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
    if 0 <= start_r < n and 0 <= start_c < n and board[start_r][start_c] == '.':
        open_ends += 1
    # 끝쪽(시퀀스 끝난 후)
    if 0 <= cur_r < n and 0 <= cur_c < n and board[cur_r][cur_c] == '.':
        open_ends += 1

    return length, open_ends

def detect_threats(board, player):
    """즉각적인 승리 위협(4목)이나 이중 위협(3-3, 4-3)을 감지합니다"""
    threats = []
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    
    # 빈 칸 탐색
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != '.':
                continue
                
            # 이 위치에 돌을 두었을 때 형성되는 패턴 확인
            temp_board = copy_board(board)
            temp_board[r][c] = player
            
            open_threes = 0
            open_fours = 0
            
            for dr, dc in directions:
                length, open_ends = count_sequence(temp_board, r, c, dr, dc, player)
                
                if length == 4 and open_ends >= 1:
                    open_fours += 1
                elif length == 3 and open_ends == 2:
                    open_threes += 1
            
            # 위협 감지
            threat_score = 0
            if open_fours > 0:  # 4목은 즉시 승리 위협
                threat_score = 1000
            elif open_threes >= 2:  # 이중 3목은 간접 승리 위협
                threat_score = 500
            elif open_threes == 1:
                threat_score = 100
                
            if threat_score > 0:
                threats.append((r, c, threat_score))
    
    return threats

def pattern_weight(length, open_ends):
    """
    길이와 열린 끝(open_ends)에 따라 가중치를 반환합니다.
    (개선된 가중치 체계)
    """
    if length >= 5:
        return 100000  # 승리 상태
    if length == 4:
        if open_ends == 2:
            return 15000   # 열린 4: 즉시 승리 위협 (강화)
        elif open_ends == 1:
            return 2000    # 닫힌 4 (강화)
    if length == 3:
        if open_ends == 2:
            return 2000    # 열린 3: 매우 강력한 수 (강화)
        elif open_ends == 1:
            return 150     # 닫힌 3 (강화)
    if length == 2:
        if open_ends == 2:
            return 150     # 열린 2 (강화)
        elif open_ends == 1:
            return 20      # 닫힌 2 (강화)
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

def evaluate_position(board, player):
    """위치 기반 평가 점수 - 중앙 근처에 돌을 두는 것을 선호"""
    score = 0
    center = BOARD_SIZE // 2
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player:
                # 중앙에 가까울수록 높은 점수
                distance = abs(r - center) + abs(c - center)
                score += max(0, 10 - distance) * 2
    return score

def evaluate_board(board, computer, opponent):
    """강화된 평가 함수"""
    # 승리 조건 체크
    if check_win(board, computer):
        return 100000
    if check_win(board, opponent):
        return -100000
    
    # 패턴 점수
    computer_pattern_score = evaluate_player(board, computer)
    opponent_pattern_score = evaluate_player(board, opponent)
    pattern_score = computer_pattern_score - opponent_pattern_score * 1.2  # 상대 패턴 가중치 증가
    
    # 위치 점수
    computer_position_score = evaluate_position(board, computer)
    opponent_position_score = evaluate_position(board, opponent)
    position_score = computer_position_score - opponent_position_score
    
    # 위협 점수
    computer_threats = detect_threats(board, computer)
    opponent_threats = detect_threats(board, opponent)
    threat_score = sum(score for _, _, score in computer_threats) - sum(score for _, _, score in opponent_threats) * 1.5
    
    # 총점
    return pattern_score + position_score * 0.5 + threat_score

def order_moves(board, moves, player, opponent):
    """유망한 수를 먼저 평가하도록 이동 순서를 최적화합니다"""
    move_scores = []
    
    for move in moves:
        r, c = move
        score = 0
        
        # 임시 보드 생성 없이 빠르게 휴리스틱 점수 계산
        
        # 1. 이 위치에 두면 이기는 경우 최우선
        temp_board = copy_board(board)
        temp_board[r][c] = player
        if check_win(temp_board, player):
            score += 100000
            move_scores.append((move, score))
            continue
            
        # 2. 상대방이 이 위치에 두면 이기는 경우 (방어) 차선
        temp_board = copy_board(board)
        temp_board[r][c] = opponent
        if check_win(temp_board, opponent):
            score += 50000
            move_scores.append((move, score))
            continue
        
        # 3. 위협 평가 (4목, 3-3 등)
        temp_board = copy_board(board)
        temp_board[r][c] = player
        
        # 간단한 패턴 점수 계산
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            length, open_ends = count_sequence(temp_board, r, c, dr, dc, player)
            score += pattern_weight(length, open_ends) * 0.5
        
        # 4. 중앙에 가까울수록 유리
        center = BOARD_SIZE // 2
        distance = abs(r - center) + abs(c - center)
        score += max(0, 10 - distance) * 5
        
        move_scores.append((move, score))
    
    # 점수 내림차순으로 정렬
    return [move for move, score in sorted(move_scores, key=lambda x: x[1], reverse=True)]

def generate_board_hash(board):
    """보드의 해시값을 반환합니다"""
    board_str = ''.join(''.join(row) for row in board)
    return hashlib.md5(board_str.encode()).hexdigest()

def alpha_beta(board, depth, alpha, beta, maximizing, start_time, time_limit, computer, opponent):
    """트랜스포지션 테이블이 적용된 알파-베타 탐색"""
    # 시간 제한 체크
    if time.time() - start_time > time_limit:
        return evaluate_board(board, computer, opponent)
    
    # 종료 조건 체크
    if depth == 0 or is_terminal(board):
        return evaluate_board(board, computer, opponent)
    
    # 트랜스포지션 테이블 키 생성
    board_hash = generate_board_hash(board)
    tt_key = (board_hash, depth, maximizing)
    
    # 테이블에 저장된 결과가 있으면 사용
    if tt_key in transposition_table:
        return transposition_table[tt_key]
    
    moves = get_legal_moves(board)
    
    # 움직임 순서 최적화
    current_player = computer if maximizing else opponent
    opponent_player = opponent if maximizing else computer
    moves = order_moves(board, moves, current_player, opponent_player)
    
    if maximizing:
        value = NEG_INFINITY
        for move in moves:
            new_board = copy_board(board)
            new_board[move[0]][move[1]] = computer
            value = max(value, alpha_beta(new_board, depth-1, alpha, beta, False, start_time, time_limit, computer, opponent))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # 가지치기
        
        # 결과 저장
        transposition_table[tt_key] = value
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
        
        # 결과 저장
        transposition_table[tt_key] = value
        return value

def iterative_deepening(board, time_limit, computer, opponent):
    """개선된 반복 심화 탐색 (시간 관리 포함)"""
    best_move = None
    depth = 1
    start_time = time.time()
    
    # 트랜스포지션 테이블 초기화
    global transposition_table
    transposition_table = {}
    
    # 위협 감지로 빠른 결정
    my_threats = detect_threats(board, computer)
    opponent_threats = detect_threats(board, opponent)
    
    # 즉시 이길 수 있는 위치가 있으면 즉시 선택
    winning_moves = [(r, c) for r, c, score in my_threats if score >= 1000]
    if winning_moves:
        return winning_moves[0]
    
    # 상대방의 승리를 막아야 하는 위치가 있으면 즉시 방어
    critical_defense = [(r, c) for r, c, score in opponent_threats if score >= 1000]
    if critical_defense:
        return critical_defense[0]
    
    # 시간 관리를 위한 예산 설정 (더 깊은 탐색에 더 많은 시간 할당)
    time_budget = {}
    total_budget = 0
    for d in range(1, 10):
        time_budget[d] = time_limit * (2**(d-1)) / (2**9 - 1)
        total_budget += time_budget[d]
    
    # 기본 후보 이동 목록 가져오기
    moves = get_legal_moves(board)
    
    # 후보가 하나뿐이면 즉시 선택
    if len(moves) == 1:
        return moves[0]
    
    # 이동 순서 최적화
    moves = order_moves(board, moves, computer, opponent)
    
    accumulated_time = 0
    
    while depth <= 9:  # 최대 깊이 제한
        depth_start_time = time.time()
        
        current_best = None
        current_best_score = NEG_INFINITY
        
        # 강화된 알파-베타 탐색으로 최선의 수 결정
        for move in moves:
            new_board = copy_board(board)
            new_board[move[0]][move[1]] = computer
            score = alpha_beta(new_board, depth - 1, NEG_INFINITY, INFINITY, False, start_time, time_limit, computer, opponent)
            
            if score > current_best_score:
                current_best_score = score
                current_best = move
            
            # 시간 초과 시 중단
            if time.time() - start_time > time_limit * 0.9:  # 안전 마진 10%
                break
        
        # 현재 깊이의 탐색이 끝나면 결과를 업데이트
        if current_best is not None:
            best_move = current_best
            
            # 승리 위치를 찾았으면 즉시 반환
            if current_best_score >= 90000:
                return best_move
        
        # 시간 관리
        depth_elapsed = time.time() - depth_start_time
        accumulated_time += depth_elapsed
        
        # 다음 깊이 탐색 가능 여부 결정
        remaining_time = time_limit - accumulated_time
        next_depth_estimate = time_budget.get(depth + 1, time_limit)
        
        depth += 1
        
        # 시간이 충분하지 않으면 중단
        if remaining_time < next_depth_estimate * 1.5:  # 안전 마진 50%
            break
        
        # 시간 제한 체크
        if time.time() - start_time > time_limit * 0.9:
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
