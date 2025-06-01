# 음성 인식기 과제 문서

## 1. 구현 설명 (Implementation Explanation)

본 프로그램은 연속 숫자 음성을 인식하기 위해 Hidden Markov Models (HMMs) 및 Viterbi 알고리즘을 기반으로 구현되었습니다.

### 1.1. 개요

음성 입력은 MFCC (Mel-frequency cepstral coefficients) 특징 벡터 시퀀스로 제공됩니다. 프로그램은 이 MFCC 시퀀스를 입력받아, 주어진 음향 모델(음소 HMM)과 언어 모델(바이그램)을 사용하여 가장 가능성이 높은 단어 시퀀스를 출력합니다.

### 1.2. 파일 파싱

인식 과정에 필요한 다양한 모델과 데이터 파일들을 파싱하는 함수들이 구현되어 있습니다 (`recognizer.py` 상단에 위치).

- **`parse_mfcc_file(filepath)`**:

  - MFCC 특징 벡터 파일 (`*.txt`)을 파싱합니다.
  - 파일의 첫 줄에는 `프레임_수 차원_수`가 명시되어 있으며, 이후 각 줄은 MFCC 벡터를 나타냅니다.
  - 결과로 MFCC 벡터들의 리스트를 반환합니다.

- **`parse_hmm_file(base_dir)`**:

  - 음소 HMM 파라미터 파일 (`hmm.txt`)을 파싱합니다.
  - 각 음소 HMM은 상태(state), 각 상태 내 가우시안 믹스처 모델(GMM)의 가중치, 평균, 분산, 그리고 상태 전이 확률 행렬로 구성됩니다.
  - 3가지 HMM 포맷(단일 가우시안, HMM 당 GMM, 상태 당 GMM)을 모두 처리할 수 있도록 구현되었습니다.
  - 음소 이름을 키로 하고, 해당 HMM 파라미터를 값으로 하는 딕셔너리를 반환합니다.

- **`parse_vocabulary_file(base_dir)`**:

  - 인식 대상 단어 목록 파일 (`vocabulary.txt`)을 파싱합니다.
  - 단어 리스트를 반환합니다.

- **`parse_dictionary_file(base_dir)`**:

  - 발음 사전 파일 (`dictionary.txt`)을 파싱합니다.
  - 각 단어에 대한 하나 이상의 음소 시퀀스 발음을 포함합니다.
  - 단어를 키로 하고, 음소 시퀀스 리스트(발음 변형을 위함)를 값으로 하는 딕셔너리를 반환합니다.

- **`parse_bigram_file(base_dir)`**:
  - 바이그램 언어 모델 파일 (`bigram.txt`)을 파싱합니다.
  - `이전_단어 다음_단어 확률` 형식으로 구성됩니다.
  - 확률 값은 로그 확률로 변환하여 저장합니다.
  - 이전 단어를 키로, (다음 단어, 로그 확률) 딕셔너리를 값으로 하는 중첩 딕셔너리 형태로 반환합니다.

### 1.3. 음향 모델 관련 함수

- **`log_gaussian_pdf(x, mean, variance)`**:

  - 주어진 관찰 벡터 `x`(MFCC 프레임), 평균 벡터 `mean`, 분산 벡터 `variance`(대각 공분산 행렬 가정)에 대해 다변수 가우시안 분포의 로그 확률 밀도 \( \log P(x | \mu, \Sigma) \)를 계산합니다.
  - 수치적 안정성을 위해 분산 플로어링(variance flooring)을 적용합니다.

- **`log_sum_exp(log_probs)`**:

  - 로그 확률들의 합을 로그 스케일에서 안정적으로 계산합니다: \( \log(\sum_i \exp(\text{log_probs}\_i)) \).
  - 언더플로우/오버플로우를 방지하는 데 사용됩니다.

- **`calculate_observation_log_likelihood(mfcc_frame, state_gmm_params)`**:
  - 특정 MFCC 프레임이 주어졌을 때, HMM의 한 상태(state)에서 해당 프레임이 관찰될 로그 가능도 \( \log P(o_t | q_t=j) \)를 계산합니다.
  - 상태의 GMM 파라미터(가중치, 평균, 분산)를 사용하여 각 가우시안 요소에 대한 `log_gaussian_pdf`를 계산하고, 이들의 가중 합을 `log_sum_exp`를 통해 구합니다.

### 1.4. 단어 HMM 구축 (`build_word_hmms`)

`phone_hmms`(음소 HMM), `dictionary`(발음 사전), `vocabulary`(단어 목록)를 입력받아 각 단어에 대한 HMM을 구축합니다.

- 각 단어는 발음 사전에 정의된 음소 시퀀스로 표현됩니다.
- 해당 음소들의 HMM을 순차적으로 연결하여 단어 HMM을 생성합니다.
- **상태 구성**: 단어 HMM의 전체 상태는 연결된 음소 HMM들의 발성 상태(emitting state)들로 구성됩니다.
- **전이 확률**:
  - **단어 시작**: 단어의 첫 번째 음소의 시작 상태로의 전이.
  - **음소 내부 전이**: 각 음소 HMM 내부의 상태 간 전이.
  - **음소 간 전이**: 한 음소의 마지막 상태에서 다음 음소의 첫 번째 상태로의 전이. 이때, 이전 음소의 종료 확률과 다음 음소의 시작 확률을 고려합니다.
  - **단어 종료**: 단어의 마지막 음소의 발성 상태에서 단어 HMM의 종료 상태로의 전이.
- 모든 전이 확률은 로그 공간에서 계산되고 저장됩니다.
- 결과로 단어 이름을 키로 하고, 해당 단어 HMM의 총 상태 수, 상태별 GMM 파라미터 리스트, 로그 전이 확률 행렬을 값으로 하는 딕셔너리를 반환합니다.

### 1.5. Viterbi 디코딩 (`viterbi_decode`)

주어진 MFCC 프레임 시퀀스에 대해 가장 가능성이 높은 단어 시퀀스를 찾는 Viterbi 알고리즘을 구현합니다.

- **입력**: MFCC 프레임 시퀀스, 구축된 단어 HMM들, 단어 목록, 바이그램 언어 모델, 언어 모델 가중치(\(\lambda_1\)), 단어 삽입 페널티(\(\lambda_2\)).
- **자료구조**:
  - `V[t][(word_idx, state_idx)]`: 시간 `t`에서 `word_idx` 단어의 `state_idx` 상태에 도달하는 최적 경로의 로그 확률.
  - `B[t][(word_idx, state_idx)]`: 시간 `t`에서 `word_idx` 단어의 `state_idx` 상태로 오게 된 이전 시간(t-1)의 최적 상태 정보 (이전 단어 인덱스, 이전 상태 인덱스, 이전 단어 이름)를 저장하는 백포인터.
- **모든 계산은 로그 공간에서 수행됩니다.**

  - 확률의 곱셈은 로그 확률의 덧셈으로, 확률의 덧셈은 `log_sum_exp` 연산으로 처리합니다.

- **1. 초기화 (시간 t=0)**:

  - 첫 번째 MFCC 프레임에 대해 각 단어 HMM의 각 가능한 시작 상태의 Viterbi 확률을 계산합니다.
  - 고려 사항:
    - 문장 시작 기호(`<s>`)에서 현재 단어로의 바이그램 언어 모델 로그 확률 (가중치 \(\lambda_1\) 적용).
    - 단어 HMM의 시작 상태에서 해당 발성 상태로의 로그 전이 확률.
    - 해당 발성 상태에서의 첫 번째 MFCC 프레임에 대한 로그 관찰 확률.
    - 단어 삽입 로그 페널티 (\(-\lambda_2\), 페널티이므로 양수 값을 빼줌).

- **2. 재귀 (시간 t=1 에서 T-1 까지)**:

  - 각 시간 `t`의 각 MFCC 프레임에 대해, 가능한 모든 단어의 모든 상태 `(curr_w_idx, curr_s_idx)`에 대해 최적 로그 확률 `V[t][(curr_w_idx, curr_s_idx)]`를 계산합니다.
  - 두 가지 종류의 전이를 고려합니다:
    - **단어 내부(Intra-word) 전이**: 동일 단어 내의 이전 상태 `(curr_w_idx, prev_s_idx_in_curr_word)`에서 현재 상태 `(curr_w_idx, curr_s_idx)`로의 전이.
      - 로그 확률 = `V[t-1][prev_state]` + `log_tp_intra` + `log_obs_prob_curr_state`.
    - **단어 간(Inter-word) 전이**: 이전 단어 `prev_w_idx`의 특정 상태 `prev_s_idx_in_prev_word`에서 현재 단어 `curr_w_idx`의 특정 상태 `curr_s_idx`로의 전이.
      - 로그 확률 = `V[t-1][prev_state]` + `log_tp_prev_emit_to_prev_end` (이전 단어 종료) + `lambda_lm_scale * log_lm_prob(prev_word | curr_word)` + `log_word_insertion_penalty` + `log_tp_curr_start_to_curr_emit` (현재 단어 시작) + `log_obs_prob_curr_state`.
  - 위 두 경우 중 더 높은 로그 확률을 선택하여 `V[t]`와 `B[t]`를 업데이트합니다.

- **3. 종료**:

  - 마지막 프레임 `T-1`에서, 각 단어의 각 발성 상태에서 해당 단어의 종료 상태로 전이하는 확률까지 고려하여 전체 경로 중 가장 높은 로그 확률을 가진 상태 `(best_final_word_idx, best_final_state_idx)`를 찾습니다.

- **4. 백트래킹**:

  - `B` 테이블을 사용하여 시간 `T-1`의 `best_final_state_key`로부터 시간 0까지 역추적하여 가장 가능성이 높은 단어 시퀀스를 재구성합니다.
  - 단어 변경이 감지될 때마다 해당 단어를 인식된 시퀀스에 추가합니다.

- **결과**: 인식된 단어들의 리스트를 반환합니다.

### 1.6. 메인 함수 (`main`) 및 출력

- **모델 로딩**: 위에서 설명한 파싱 함수들을 호출하여 모든 필요한 모델(음소 HMM, 사전, 어휘, 바이그램 LM)을 로드합니다.
- **단어 HMM 구축**: 로드된 음소 HMM과 사전을 사용하여 단어 HMM들을 구축합니다.
- **MFCC 파일 처리**: `reference.txt` 파일을 읽어 처리할 MFCC 파일 목록을 가져옵니다. 각 MFCC 파일에 대해 다음을 수행합니다:
  - MFCC 데이터 파싱.
  - `viterbi_decode` 함수를 호출하여 인식된 단어 시퀀스를 얻습니다.
  - 결과를 `all_recognition_results` 딕셔너리에 저장합니다 (키: 논리적 경로, 값: 단어 리스트).
- **결과 저장 (`write_mlf_output`)**: 모든 파일 처리가 끝나면 `all_recognition_results`를 `OUTPUT_MLF_FILE_PATH` (기본값: `recognized.txt`)에 MLF (Master Label File) 형식으로 저장합니다.
  - MLF 형식: `#MLF!#` 헤더로 시작, 각 파일에 대해 `"<logical_path>"`와 인식된 단어들이 줄 단위로 기록되고 마침표(`.`)로 끝납니다.
- **성능 평가**:
  - `subprocess` 모듈을 사용하여 `HResults.exe` (Wine을 통해 실행)를 호출합니다.
  - `HResults.exe`는 `reference.txt`(정답)와 생성된 `recognized.txt`(인식 결과)를 비교하여 성능 지표(정확도, 오류율: Substitution, Deletion, Insertion) 및 혼동 행렬(Confusion Matrix)을 계산하고 콘솔에 출력합니다.

## 2. 프로그램 실행 방법 안내 (Program Execution Guide)

### 2.1. 요구 사항

- **Python 3.x**
- **NumPy 라이브러리**: 설치되지 않은 경우, `pip install numpy` 명령어로 설치합니다.
- **(macOS/Linux에서 `HResults.exe` 실행 시) Wine**: Windows 실행 파일인 `HResults.exe`를 macOS 또는 Linux 환경에서 실행하기 위해 필요합니다. Homebrew를 사용하는 경우 `brew install --cask wine-stable` 명령어로 설치할 수 있습니다. Wine 설치 후 Gatekeeper 설정 문제 발생 시, 터미널에서 `sudo xattr -cr "/Applications/Wine Stable.app"` 명령어를 실행하여 해결할 수 있습니다.

### 2.2. 디렉토리 구조

정상적인 실행을 위해 다음과 같은 디렉토리 구조를 유지해야 합니다:

```
your_workspace_root/
└── assignment/
    └── final-assignment/
        ├── recognizer.py       <-- 실행 스크립트
        ├── ASR_Homework-1/     <-- 과제 제공 파일들이 있는 폴더
        │   ├── hmm.txt
        │   ├── dictionary.txt
        │   ├── vocabulary.txt
        │   ├── bigram.txt
        │   ├── reference.txt
        │   ├── HResults.exe
        │   └── mfc/              <-- MFCC 파일들이 있는 폴더
        │       ├── f/
        │       │   ├── ak/
        │       │   │   └── *.txt
        │       │   └── ...
        │       └── m/
        │           ├── ak/
        │           │   └── *.txt
        │           └── ...
        ├── recognized.txt      <-- (실행 후 생성됨) 인식 결과 MLF 파일
        ├── HResults-Output.txt <-- (실행 후 생성됨, 직접 저장 시) HResults.exe 출력
        └── docs.md             <-- 이 문서
```

- `recognizer.py` 스크립트는 `ASR_Homework-1` 폴더와 같은 `final-assignment` 디렉토리 내에 있어야 합니다.
- `ASR_Homework-1` 폴더에는 과제 설명에 명시된 모든 입력 파일(`hmm.txt`, `dictionary.txt` 등)과 `HResults.exe`, 그리고 `mfc/` 하위 디렉토리에 모든 MFCC 파일(`*.txt`)들이 포함되어 있어야 합니다.

### 2.3. 실행 명령어

1.  터미널 또는 명령 프롬프트를 엽니다.
2.  스크립트가 위치한 디렉토리의 상위 디렉토리로 이동합니다. 예를 들어, 워크스페이스 루트가 `/Users/jeongseogmin/Desktop/AI` 라면:
    ```bash
    cd /Users/jeongseogmin/Desktop/AI
    ```
3.  다음 명령어를 사용하여 `recognizer.py` 스크립트를 실행합니다:
    ```bash
    python assignment/final-assignment/recognizer.py
    ```
    또는 `final-assignment` 디렉토리로 직접 이동한 경우:
    ```bash
    cd assignment/final-assignment
    python recognizer.py
    ```

### 2.4. 실행 결과

- 스크립트가 실행되면, 각 MFCC 파일 처리 과정 및 인식된 단어 시퀀스가 콘솔에 출력됩니다.
- 모든 파일 처리가 완료되면, 인식 결과가 `assignment/final-assignment/recognized.txt` 파일에 MLF 형식으로 저장됩니다.
- 이후, `HResults.exe`가 자동으로 호출되어 `reference.txt`와 `recognized.txt`를 비교 분석한 결과 (문장/단어 정확도, 오류율, 혼동 행렬 등)가 콘솔에 출력됩니다.
  - 이 콘솔 출력은 `assignment/final-assignment/HResults-Output.txt` 파일에도 기록되어 있습니다 (사용자가 직접 해당 파일로 출력을 리디렉션하거나 복사/붙여넣기 한 경우).

### 2.5. 하이퍼파라미터 튜닝

`recognizer.py` 파일 내의 `main` 함수 상단에서 다음 변수들의 값을 조정하여 인식 성능을 튜닝할 수 있습니다:

- **`lambda_lm_scale_val`**: 언어 모델 점수에 곱해지는 가중치입니다. 이 값을 조정하여 언어 모델의 영향력을 조절할 수 있습니다. (기본값: `10.0`)
- **`word_insertion_penalty_val`**: 단어가 인식될 때마다 적용되는 페널티 값입니다. 양수 값으로 설정되며, 로그 공간에서는 이 값에 음수를 취한 값이 더해집니다 (즉, 페널티가 클수록 로그 확률이 낮아짐). 이 값을 조정하여 인식되는 단어의 평균 길이를 제어할 수 있습니다. (기본값: `2.5`)

이 값들을 변경하며 `HResults.exe`의 출력(특히 Insertion/Deletion 오류율)을 관찰하여 최적의 조합을 찾는 것이 좋습니다.
