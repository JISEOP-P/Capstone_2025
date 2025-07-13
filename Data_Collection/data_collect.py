import os
import sys
import time
import numpy as np
import keyboard
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp

# ---------------------------------------------------
# OS별 stdin flush 함수
# ---------------------------------------------------
if os.name == 'nt':
    import msvcrt
    def flush_stdin():
        while msvcrt.kbhit():
            msvcrt.getch()
else:
    import select
    def flush_stdin():
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)

def flush_keyboard():
    while keyboard.is_pressed('enter') or keyboard.is_pressed('space'):
        time.sleep(0.05)

# ---------------------------------------------------
# Radar config
# ---------------------------------------------------
RADAR_CONFIG = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=0.08,
    chirp_repetition_time_s=300e-6,
    num_chirps=256,
    tdm_mimo=False,
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=58e9,
        end_frequency_Hz=63.5e9,
        sample_rate_Hz=1e6,
        num_samples=128,
        rx_mask=7,
        tx_mask=1,
        tx_power_level=31,
        lp_cutoff_Hz=500000,
        hp_cutoff_Hz=80000,
        if_gain_dB=25,
    ),
)

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def get_input(prompt, valid_set, transform=str):
    while True:
        val = input(prompt).strip()
        if transform(val) in valid_set:
            return transform(val)
        else:
            print(f"[Error] 잘못된 입력: '{val}'. 다시 시도하세요.")

def get_input_range(prompt, min_val, max_val, padding=3):
    while True:
        val = input(prompt).strip()
        if val.isdigit() and min_val <= int(val) <= max_val:
            return val.zfill(padding)
        else:
            print(f"[Error] 잘못된 입력: '{val}'. {min_val}~{max_val} 사이의 숫자를 입력하세요.")

def build_filename(team_code, label, s_env, v_env, data_num, subject_code):
    return f"{team_code}_{label}_s{s_env}_v{v_env}_{data_num}_{subject_code}.npy"

# ---------------------------------------------------
# Main loop
# ---------------------------------------------------
def main():
    os.makedirs("Data_Collection", exist_ok=True)

    print("▶ 레이더 초기화 중...")
    with DeviceFmcw() as device:
        print("Radar SDK Version:", get_version())
        print("UUID of board:", device.get_board_uuid())
        print("Sensor:", device.get_sensor_type())

        sequence = device.create_simple_sequence(RADAR_CONFIG)
        device.set_acquisition_sequence(sequence)
        print("▶ 레이더 설정 완료")

        team_code = "GDN"
        subject_code = get_input("피실험자 고유코드 (G, D, N): ", {"G", "D", "N", "g", "d", "n"}, str.upper)
        label = get_input("라벨 번호 (0~14): ", set(map(str, range(15))))
        s_env = get_input("데이터 수집 환경 s(1~4,A,B,C): ", {"1", "2", "3", "4", "A", "a", "B", "b", "C", "c"}, str.upper)
        v_env = get_input("세부 수집 세팅 v(1~4): ", {"1", "2", "3", "4"})
        data_num = get_input_range("데이터 번호 (001~030): ", 1, 30)

        while True:
            filename = build_filename(team_code, label, s_env.upper(), v_env, data_num, subject_code)
            file_path = os.path.join("Data_Collection", filename)
            print(f"\n[대기 상태] 현재 설정 -> {filename}")
            print("스페이스바: 데이터 수집 | 엔터: 파일명 설정 변경 | ESC: 종료")

            event = keyboard.read_event()
            while event.event_type != 'down':
                event = keyboard.read_event()
            key = event.name

            if key == 'space':
                print("▶ 데이터 수집 시작...")
                frames_list = []
                start_time = time.time()
                for frame_number in range(50):
                    frame_contents = device.get_next_frame()
                    for frame in frame_contents:
                        frames_list.append(frame)
                    elapsed_now = time.time() - start_time
                    print(f"\r[Progress] Frame {frame_number+1}/50 - elapsed {elapsed_now:.2f}s", end='')
                print()

                # stop acquisition
                device.stop_acquisition()
                time.sleep(0.5)

                all_frames = np.stack(frames_list, axis=0)
                np.save(file_path, all_frames)
                print(f"[Info] 저장 완료: {file_path} | shape: {all_frames.shape}")
                print("[Info] 측정 소요 시간:", f"{time.time() - start_time:.2f}초")

                # acquisition 재시작
                device.set_acquisition_sequence(sequence)
                time.sleep(0.5)

                # 데이터 번호 증가
                next_num = int(data_num) + 1
                if next_num > 30:
                    print(f"\n[s{s_env.upper()} 수집 완료] Enter key를 누르세요.")
                    while True:
                        if keyboard.read_event().name == 'enter':
                            flush_keyboard()
                            flush_stdin()
                            break
                    data_num = "030"  # 그대로 고정
                else:
                    data_num = str(next_num).zfill(3)

            elif key == 'enter':
                flush_keyboard()
                flush_stdin()
                print("\n▶ 파일명 재설정 ([기존값], 변경하려면 수정 후 enter)")

                while True:
                    tmp_input = input(f"피실험자 고유코드 (G, D, N) [{subject_code}]: ").strip()
                    if tmp_input == "":
                        break
                    tmp_input = tmp_input.upper()
                    if tmp_input in {"G", "D", "N"}:
                        subject_code = tmp_input
                        break
                    else:
                        print("잘못된 입력. G,D,N 중 입력하세요.")

                while True:
                    tmp_input = input(f"라벨 번호 (0~14) [{label}]: ").strip()
                    if tmp_input == "":
                        break
                    if tmp_input in set(map(str, range(15))):
                        label = tmp_input
                        break
                    else:
                        print("잘못된 입력. 0~14 중 입력하세요.")

                while True:
                    tmp_input = input(f"데이터 수집 환경 s(1~4,A,B,C) [{s_env}]: ").strip()
                    if tmp_input == "":
                        break
                    if tmp_input.upper() in {"1", "2", "3", "4", "A", "B", "C"}:
                        s_env = tmp_input.upper()
                        break
                    else:
                        print("잘못된 입력. 1~4,A,B,C 중 입력하세요.")

                while True:
                    tmp_input = input(f"세부 수집 세팅 v(1~4) [{v_env}]: ").strip()
                    if tmp_input == "":
                        break
                    if tmp_input in {"1", "2", "3", "4"}:
                        v_env = tmp_input
                        break
                    else:
                        print("잘못된 입력. 1~4 중 입력하세요.")

                while True:
                    tmp_input = input(f"데이터 번호 (001~030) [{data_num}]: ").strip()
                    if tmp_input == "":
                        break
                    if tmp_input.isdigit() and 1 <= int(tmp_input) <= 30:
                        data_num = tmp_input.zfill(3)
                        break
                    else:
                        print("잘못된 입력. 001~030 중 입력하세요.")

                flush_keyboard()
                flush_stdin()

            elif key == 'esc':
                print("[Esc] 종료합니다.")
                break

if __name__ == "__main__":
    main()
