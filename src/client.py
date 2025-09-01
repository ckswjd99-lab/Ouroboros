'''
client.py

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
import torch
from multiprocessing import Process, Queue, Semaphore

from typing import List, Tuple

from networking import (
    HEADER_NORMAL, HEADER_TERMINATE,
    connect_dual_tcp, transmit_data, receive_data,
    measure_timelag,
    int32_to_bytes, bytes_to_int32,
    bool_to_bytes, bytes_to_bool
)
from preprocessing import (
    ndarray_to_bytes, bytes_to_ndarray, load_video,
    estimate_affine_in_padded_anchor,
    estimate_affine_in_padded_anchor_fast,
    apply_affine_and_pad, get_padded_image,
    create_dirtiness_map,
    # create_dirtiness_map_percentage,
    rigid_from_mvs
)

from custom_h264 import custom_h264


def thread_send_video(
    socket_tx: socket.socket, 
    video_path: str, 
    frame_rate: float, 
    sem_offload, 
    queue_preproc_timestamp: Queue,
    queue_timestamp: Queue, 
    compress: str,
) -> float:
    """
    ① 원본 프레임 → anchor_image_padded·dirtiness_map 등 생성
    ② 생성된 결과만 서버로 전송
    """
    
    PAD_FILLING_COLOR = (0, 0, 0)
    
    frames = load_video(video_path)
    WIDTH, HEIGHT = frames[0].shape[1], frames[0].shape[0]

    input_size = (1024, 1024)
    block_size = 16
    frame_shape = (WIDTH, HEIGHT)
    gop = int(frame_rate)
    fw, fh = frame_shape
    center_vec = np.array([input_size[1] / 2 - fw / 2,
                           input_size[0] / 2 - fh / 2], dtype=np.float32)
    prev_H = np.array([[1, 0, center_vec[0]],
                       [0, 1, center_vec[1]],
                       [0, 0, 1]], dtype=np.float32)
    scaling_factor = 1.0

    anchor_image_padded = None

    if compress == "h264":
        # X264_PARAMS = "keyint=9999:min-keyint=9999:no-scenecut=1:bframes=0:" \
        #               "ref=1:deadzone-inter=1000:deadzone-intra=300:" \
        #               "aq-mode=0:psy-rd=0.0:mbtree=0:rc-lookahead=0:qpstep=10"
        X264_PARAMS = None
        custom_h264.mv_init(WIDTH, HEIGHT, int(frame_rate), X264_PARAMS)

        anchor_codec = av.codec.CodecContext.create("libx264", "w")
        anchor_codec.width  = input_size[1]          # W,H 순서 주의
        anchor_codec.height = input_size[0]
        anchor_codec.pix_fmt = "yuv420p"
        # 빠른 단일-프레임 인코딩 옵션
        anchor_codec.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "23"}
        anchor_codec.open()


    for fidx, frame in enumerate(frames):
        if sem_offload is not None:
            sem_offload.acquire()

        # ---------- (1) 타임스탬프 ----------
        queue_timestamp.put(time.time())

        # ---------- (2) 전처리(기존 server 코드 그대로) ----------
        target_ndarray = cv2.resize(frame, frame_shape)
        refresh = False
        shift_x, shift_y = 0, 0

        if fidx % gop == 0 or anchor_image_padded is None:
            refresh = True
        else:
            if compress == "h264":
                _, mvs = custom_h264.mv_process_and_encode(frame)
                M_cur_prev = rigid_from_mvs(mvs)
                if M_cur_prev is not None:
                    H_cur_prev = np.vstack([M_cur_prev, [0, 0, 1]])
                    cum_H = prev_H @ H_cur_prev
                    prev_H = cum_H.copy()
                    affine_matrix = cum_H[:2, :].copy()
                else:
                    refresh = True
                    affine_matrix = None
            else:
                affine_matrix = estimate_affine_in_padded_anchor_fast(
                    anchor_image_padded, target_ndarray)
                
            if affine_matrix is not None:
                target_padded_ndarray, (shift_x, shift_y), affine_matrix = \
                    apply_affine_and_pad(target_ndarray, affine_matrix,
                                         block_size=block_size,
                                         filling_color=PAD_FILLING_COLOR)
                prev_H = np.vstack([affine_matrix, [0, 0, 1]])
                anchor_image_padded = anchor_image_padded.copy()
                anchor_image_padded = np.roll(anchor_image_padded,
                                              shift=(-shift_x, -shift_y),
                                              axis=(1, 0))

                if target_padded_ndarray is not None:
                    scaling_factor = np.linalg.norm(affine_matrix[:2, :2])
                    dirtiness_map = create_dirtiness_map(
                        anchor_image_padded, target_padded_ndarray)
                    # dirtiness_map = create_dirtiness_map_percentage(
                    #     anchor_image_padded, target_padded_ndarray)
                else:
                    target_padded_ndarray = get_padded_image(
                        target_ndarray, input_size,
                        filling_color=PAD_FILLING_COLOR)
                    dirtiness_map = torch.ones(1, 64, 64, 1)
                    scaling_factor = 1.0
                    prev_H = np.array([[1, 0, center_vec[0]],
                                       [0, 1, center_vec[1]],
                                       [0, 0, 1]], dtype=np.float32)
            refresh |= (target_padded_ndarray is None)
            refresh |= (scaling_factor < 0.95)
            # print(f"scaling_factor: {scaling_factor:.2f}")
            refresh |= (scaling_factor > 1.7)

        if refresh:
            dirtiness_map = torch.ones(1, 64, 64, 1)
            target_padded_ndarray = get_padded_image(
                target_ndarray, input_size,
                filling_color=PAD_FILLING_COLOR)
            prev_H = np.array([[1, 0, center_vec[0]],
                               [0, 1, center_vec[1]],
                               [0, 0, 1]], dtype=np.float32)

        dmap_np = dirtiness_map.cpu().numpy()
        dmap_resized = cv2.resize(dmap_np[0, :, :, 0],
                                  (input_size[1], input_size[0]))
        dmap_resized = np.expand_dims(dmap_resized, axis=-1)
        dmap_resized = np.concatenate([dmap_resized] * 3, axis=-1)

        if anchor_image_padded is None:
            anchor_image_padded = target_padded_ndarray
        else:
            anchor_image_padded = dmap_resized * target_padded_ndarray + \
                (1 - dmap_resized) * anchor_image_padded
            anchor_image_padded = anchor_image_padded.astype(np.uint8)

        queue_preproc_timestamp.put(time.time())

        # ---------- (3) 전송 ----------
        transmit_data(socket_tx, int32_to_bytes(fidx))
        transmit_data(socket_tx, bool_to_bytes(refresh))
        transmit_data(socket_tx, ndarray_to_bytes(dmap_np))
        transmit_data(socket_tx, int32_to_bytes(shift_x // block_size))
        transmit_data(socket_tx, int32_to_bytes(shift_y // block_size))
        transmit_data(socket_tx, int32_to_bytes(block_size))
        transmit_data(socket_tx, ndarray_to_bytes(prev_H))

        if compress == "none":
            transmit_data(socket_tx, ndarray_to_bytes(anchor_image_padded))

        elif compress == "jpeg":
            # JPEG - 90% 품질
            ok, enc = cv2.imencode(".jpg", anchor_image_padded,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            transmit_data(socket_tx, enc.tobytes())

        elif compress == "h264":
            vf = av.VideoFrame.from_ndarray(anchor_image_padded, format="bgr24")
            
            pkts = anchor_codec.encode(vf) or []
            transmit_data(socket_tx, int32_to_bytes(len(pkts)))

            for p in pkts:
                transmit_data(socket_tx, bytes(p))

            # buffer for the next frame
            bf = av.VideoFrame.from_ndarray(target_padded_ndarray, format="bgr24")

            pkts = anchor_codec.encode(bf) or []
            transmit_data(socket_tx, int32_to_bytes(len(pkts)))

            for p in pkts:
                transmit_data(socket_tx, bytes(p))

        if sem_offload is not None:
            pass

    if compress == "h264":
        custom_h264.mv_close()

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
    annotation_path = args.annotation_path
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
    queue_preproc_timestamp = Queue(maxsize=10000)
    queue_transmit_timestamp = Queue(maxsize=10000)
    queue_receive_timestamp = Queue(maxsize=10000)

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
            queue_preproc_timestamp,
            queue_transmit_timestamp,
            compress
        )
    )

    # Load annotations if needed
    if annotation_path:
        try:
            import json
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            print(f"Loaded annotations from {annotation_path}")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            annotations = None
    else:
        annotations = None
        print("No annotations provided.")

    # Send metadata
    metadata = {
        "gop": gop,
        "compress": compress,
        "frame_shape": (854, 480),
        "video_basename": video_path.split('/')[-1].strip('.mp4'),
    }
    if annotations:
        metadata["annotations"] = annotations
    metadata_bytes = str(metadata).encode('utf-8')
    transmit_data(socket_tx, metadata_bytes)

    # print("Sent metadata:", metadata)

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
    preproc_times = []
    receive_times = []
    while not queue_transmit_timestamp.empty():
        transmit_times.append(queue_transmit_timestamp.get())
    while not queue_preproc_timestamp.empty():
        preproc_times.append(queue_preproc_timestamp.get())
    while not queue_receive_timestamp.empty():
        receive_times.append(queue_receive_timestamp.get())
    transmit_times = np.array(transmit_times)
    preproc_times = np.array(preproc_times)
    receive_times = np.array(receive_times)

    transmit_data(socket_tx, ndarray_to_bytes(transmit_times))
    transmit_data(socket_tx, ndarray_to_bytes(preproc_times))
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

    DEFAULT_SEQUENCE = "drone"

    parser = argparse.ArgumentParser(description="Client for sending video to server.")
    parser.add_argument("--server-ip", type=str, default="localhost", help="Server IP address.")
    parser.add_argument("--server-port", type=int, default=65432, help="Server port.")
    parser.add_argument("--video-path", type=str, default=f"/data/DAVIS2017_trainval/MP4Videos/480p/{DEFAULT_SEQUENCE}.mp4", help="Path to the video file.")
    # parser.add_argument("--video-path", type=str, default=f"/data/vid_data/vid/vid_val/videos/00005001.mp4", help="Path to the video file.")
    parser.add_argument("--annotation-path", type=str, default=f"/data/DAVIS2017_trainval/Annotations_bbox/480p/{DEFAULT_SEQUENCE}.json", help="Path to the annotation file.")
    # parser.add_argument("--annotation-path", type=str, default=None, help="Path to the annotation file.")
    parser.add_argument("--frame-rate", type=float, default=100, help="Frame rate for sending video.")
    parser.add_argument("--sequential", type=bool, default=True, help="Sender waits until the result of the previous frame is received.")
    parser.add_argument("--compress", type=str, default="h264", help="Compress video frames before sending.")

    args = parser.parse_args()

    main(args)