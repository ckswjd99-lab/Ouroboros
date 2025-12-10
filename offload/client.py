'''
device.py

1) connect to server
2) load video and send it to the model
3) receive the result

'''

import cv2
import numpy as np
import socket
import time
import av
import sys
from multiprocessing import Process, Queue, Semaphore

from typing import List, Tuple

from networking import (
    HEADER_NORMAL, HEADER_TERMINATE,
    connect_dual_tcp, transmit_data, receive_data,
    measure_timelag,
)
from preprocessing import ndarray_to_bytes, bytes_to_ndarray, load_video

from custom_h264 import custom_h264


def rigid_from_mvs(mvs: np.ndarray):
    """
    모션벡터(N, 5) → 현재 프레임 ➜ 참조(이전) 프레임으로 가는 2×3 rigid 변환 추정.
    실패 시 None 반환.
    """
    if mvs is None or mvs.shape[0] < 3 or mvs.shape[1] != 5:
        return None
    # mvs: [dst_x, dst_y, motion_x, motion_y, motion_scale]
    dst = mvs[:, 0:2].astype(np.float32)
    src = dst + (mvs[:, 2:4] / mvs[:, 4:5]).astype(np.float32)
    if dst.shape[0] < 3 or src.shape[0] < 3:
        return None
    M, _ = cv2.estimateAffinePartial2D(dst, src,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=2.0,
                                       confidence=0.995,
                                       refineIters=10)
    return M                                                   # shape (2,3) or None


def thread_send_video(
    socket_tx: socket.socket, 
    video_path: str, 
    frame_rate: float, 
    sem_offload, 
    queue_timestamp: Queue, 
    compress: str,
) -> float:
    """
    Thread function to send video frames to the server.

    Args:
        socket (socket.socket): The socket object to send data over.
        video_path (str): Path to the video file.
        frame_rate (float): Frame rate for sending video frames.
        sem_offload (Semaphore): Semaphore for sequential offloading.
        queue_timestamp (Queue): Queue to store timestamps.
        compress (str): Compression method to use ('jpeg', 'h264', or 'none').
    """
    frames = load_video(video_path)
    WIDTH, HEIGHT = frames[0].shape[1], frames[0].shape[0]

    if compress == "h264":
        X264_PARAMS = "keyint=9999:min-keyint=9999:no-scenecut=1:bframes=0:ref=1:deadzone-inter=1000:deadzone-intra=300:aq-mode=0:psy-rd=0.0:mbtree=0:rc-lookahead=0:qpstep=10"
        custom_h264.mv_init(WIDTH, HEIGHT, int(frame_rate), X264_PARAMS)

    for fidx, frame in enumerate(frames):
        if sem_offload is not None:
            sem_offload.acquire()  # Wait for server to process the frame

        # Transmit frame idx
        fidx_bytes = fidx.to_bytes(4, 'big')
        transmit_data(socket_tx, fidx_bytes)

        timestamp = time.time()

        # Optionally encode the frame
        if compress == "jpeg":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, frame_enc = cv2.imencode('.jpg', frame, encode_param)
            frame_enc = np.array(frame_enc)
            frame_bytes = ndarray_to_bytes(frame_enc)
            transmit_data(socket_tx, frame_bytes)
        
        elif compress == "h264":
            # mv_lib로 인코딩 및 모션벡터 추출
            encoded_bytes, mvs = custom_h264.mv_process_and_encode(frame)
            # 인코딩 바이트 전송 (길이 포함)
            transmit_data(socket_tx, encoded_bytes)
            # 모션벡터 전송
            M_cur_prev = rigid_from_mvs(mvs)
            if M_cur_prev is None:
                M_cur_prev = -np.ones((2, 3), dtype=np.float32)  # Fallback to identity if estimation fails

            mv_bytes = ndarray_to_bytes(M_cur_prev)
            transmit_data(socket_tx, mv_bytes)
        
        else:
            frame_bytes = ndarray_to_bytes(frame)
            transmit_data(socket_tx, frame_bytes)

        # Logging
        print(f"Transferred {fidx} | frame: {frame.shape} | timestamp: {timestamp}")
        queue_timestamp.put(timestamp)

    if compress == "h264":
        custom_h264.mv_close()

    # Send termination signal
    transmit_data(socket_tx, b"", HEADER_TERMINATE)


def thread_receive_results(
    socket: socket.socket, 
    sem_offload, 
    queue_timestamp: Queue
) -> float:
    """
    Thread function to receive results from the server.

    Args:
        socket (socket.socket): The socket object to receive data from.
    """
    while True:
        data = receive_data(socket)
        if data is None:
            break
        
        fidx = int.from_bytes(data[:4], 'big')
        boxes = bytes_to_ndarray(receive_data(socket))
        scores = bytes_to_ndarray(receive_data(socket))
        labels = bytes_to_ndarray(receive_data(socket))

        timestamp = time.time()
        print(f"Received {fidx} | boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape} | timestamp: {timestamp}")
        queue_timestamp.put(timestamp)

        if sem_offload is not None:
            sem_offload.release()  # Release semaphore to indicate processing is done
        

def main(args):
    server_ip = args.server_ip
    server_port1 = args.server_port
    server_port2 = server_port1 + 1
    video_path = args.video_path
    gop = args.frame_rate
    sequential = args.sequential
    compress = args.compress

    # Connect to the server
    socket_rx, socket_tx = connect_dual_tcp(
        server_ip, (server_port1, server_port2), node_type="client"
    )

    timelag = measure_timelag(socket_rx, socket_tx, "client")
    print(f"Timelag: {timelag} seconds")

    # Create queues for recording
    queue_transmit_timestamp = Queue(maxsize=100)
    queue_receive_timestamp = Queue(maxsize=100)

    # Create a semaphore for offloading
    sem_offload = Semaphore(1) if sequential else None

    # Prepare processes
    thread_recv = Process(
        target=thread_receive_results, 
        args=(
            socket_rx, 
            sem_offload, 
            queue_receive_timestamp
        )
    )
    thread_send = Process(
        target=thread_send_video, 
        args=(
            socket_tx, 
            video_path, 
            gop, 
            sem_offload, 
            queue_transmit_timestamp,
            compress
        )
    )

    # Send metadata
    metadata = {
        "gop": gop,
        "compress": compress,
        "frame_shape": (854, 480)
    }
    metadata_bytes = str(metadata).encode('utf-8')
    transmit_data(socket_tx, metadata_bytes)

    print("Sent metadata:", metadata)

    # Wait for server to be ready
    receive_data(socket_rx)

    # Start sending video
    thread_recv.start()
    thread_send.start()

    # Wait for the threads to finish
    thread_send.join()
    thread_recv.join()

    # Send statistics
    transmit_times = []
    receive_times = []
    while not queue_transmit_timestamp.empty():
        transmit_times.append(queue_transmit_timestamp.get())
    while not queue_receive_timestamp.empty():
        receive_times.append(queue_receive_timestamp.get())
    transmit_times = np.array(transmit_times)
    receive_times = np.array(receive_times)

    transmit_data(socket_tx, ndarray_to_bytes(transmit_times))
    transmit_data(socket_tx, ndarray_to_bytes(receive_times))

    # Close sockets
    socket_rx.close()
    socket_tx.close()

    # Print the stats

    latencies = [receive - transmit for transmit, receive in zip(transmit_times, receive_times)]
    print(f"Average latency: {np.mean(latencies):.4f} seconds")
    print(f"Max latency: {np.max(latencies):.4f} seconds")
    print(f"Min latency: {np.min(latencies):.4f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client for sending video to server.")
    parser.add_argument("--server-ip", type=str, default="localhost", help="Server IP address.")
    parser.add_argument("--server-port", type=int, default=65432, help="Server port.")
    parser.add_argument("--video-path", type=str, default="./input.mp4", help="Path to the video file.")
    parser.add_argument("--frame-rate", type=float, default=100, help="Frame rate for sending video.")
    parser.add_argument("--sequential", type=bool, default=True, help="Sender waits until the result of the previous frame is received.")
    parser.add_argument("--compress", type=str, default="h264", help="Compress video frames before sending.")

    args = parser.parse_args()

    main(args)