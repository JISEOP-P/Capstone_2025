# ===========================================================================
# [1] 모듈 import
# ===========================================================================

import pprint
import numpy as np
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
import time

# ===========================================================================
# [2] 센서 설정
# ===========================================================================

config = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=0.08,  # Frame repetition time
    chirp_repetition_time_s=300e-6,  # Chirp repetition time
    num_chirps=256,  # chirps per frame
    tdm_mimo=False,  # set True to enable MIMO mode, which is only valid for sensors with 2 Tx antennas
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=58e9,  # start RF frequency, where Tx is ON
        end_frequency_Hz=63.5e9,  # stop RF frequency, where Tx is OFF
        sample_rate_Hz=1e6,  # ADC sample rate
        num_samples=128,  # samples per chirp
        rx_mask=7,  # RX mask is a 4-bit, each bit set enables that RX e.g. [1,3,7,15]
        tx_mask=1,  # TX antenna mask is a 2-bit (use value 3 for MIMO)
        tx_power_level=31,  # TX power level of 31
        lp_cutoff_Hz=500000,  # Anti-aliasing filter cutoff frequency, select value from data-sheet
        hp_cutoff_Hz=80000,  # High-pass filter cutoff frequency, select value from data-sheet
        if_gain_dB=28,  # IF-gain
    ),
)

# =========================================================================== 
# [3] 데이터 수집 (실시간 진행상황 표시)
# =========================================================================== 

with DeviceFmcw() as device:
    print("Radar SDK Version: " + get_version())
    print("UUID of board: " + device.get_board_uuid())
    print("Sensor: " + str(device.get_sensor_type()))
    pp = pprint.PrettyPrinter()

    sequence = device.create_simple_sequence(config)
    device.set_acquisition_sequence(sequence)

    chirp_loop = sequence.loop.sub_sequence.contents
    metrics = device.metrics_from_sequence(chirp_loop)
    pp.pprint(metrics)

    pp.pprint(create_dict_from_sequence(sequence))
    device.save_register_file("exported_registers.txt")

    frames_list = []

    print("Start collecting frames...")
    start_time = time.time()

    for frame_number in range(50):
        frame_contents = device.get_next_frame()

        for frame in frame_contents:
            frames_list.append(frame)

        elapsed_now = time.time() - start_time
        print(f"\r[Progress] Frame {frame_number+1}/50 - elapsed {elapsed_now:.2f}s", end='')

    print()  # 줄 바꿈
    total_elapsed = time.time() - start_time

    all_frames = np.stack(frames_list, axis=0)
    np.save("radar_4s_50frames.npy", all_frames)
    print(f"Saved shape: {all_frames.shape} to radar_5s_50frames.npy")
    print(f"[Info] 측정 소요 시간: {total_elapsed:.2f}초")

