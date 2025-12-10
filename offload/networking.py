import socket
import time

from typing import Optional, Tuple

# TRANSMISSION HEADER
HEADER_NORMAL = 0
HEADER_TERMINATE = 1



def connect_dual_tcp(
    host: str,
    ports: Tuple[int, int] = (65432, 65433),
    node_type: str = "server"
) -> Tuple[socket.socket, socket.socket]:
    """
    Connect to a dual TCP socket.

    If the type is 'server', it creates two server sockets and waits for a client to connect.
    If the type is 'client', it repeatedly tries to connect to the server sockets.

    When one connection is established, it continues to connect to the other socket.
    As a result, two connections are established, one for each port.

    Args:
        host (str): The hostname or IP address to connect to.
        ports (Tuple[int, int]): A tuple containing two port numbers.
        type (str): The type of connection ('server' or 'client').

    Returns:
        Tuple[socket.socket, socket.socket]: A tuple containing two connected socket objects.
        (RX, TX)

    """

    if node_type == "server":
        # Create server sockets
        server_sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in ports]
        for server_socket, port in zip(server_sockets, ports):
            server_socket.bind((host, port))
            server_socket.listen(1)
            print(f"Server listening on {host}:{port}")

        # Accept connections
        connections = []
        for server_socket in server_sockets:
            conn, addr = server_socket.accept()
            print(f"Connected to {addr}")
            connections.append(conn)

        return tuple(connections)

    elif node_type == "client":
        # Create client sockets
        client_sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in ports]
        for client_socket, port in zip(client_sockets, ports):
            while True:
                print(f"Trying to connect to {host}:{port}")
                try:
                    client_socket.connect((host, port))
                    print(f"Connected to {host}:{port}")
                    break
                except socket.error as e:
                    print(f"Connection failed: {e}. Retrying...")

                    time.sleep(1)  # Wait before retrying
                    continue

        return tuple(client_sockets[::-1])
    else:
        raise ValueError("Invalid type. Use 'server' or 'client'.")

def transmit_data(socket: socket.socket, data: bytes, header: int=HEADER_NORMAL) -> None:
    """
    Transmit data over a socket.

    Args:
        socket (socket.socket): The socket object to transmit data over.
        data (bytes): The data to transmit.

    """
    header = header.to_bytes(4, 'big')  # Convert header to bytes
    data_length = len(data)
    
    socket.sendall(header)  # Send the header
    socket.sendall(data_length.to_bytes(4, 'big'))  # Send the length of the data
    socket.sendall(data)  # Send the actual data

def receive_data(socket: socket.socket, buffer_size: int = 1024) -> bytes | None:
    """
    Receive data from a socket.
    Ensures that buffer_size is received.

    Args:
        socket (socket.socket): The socket object to receive data from.
        buffer_size (int): The maximum amount of data to receive at once.

    Returns:
        bytes: The received data.

    """
    data_header = socket.recv(4)  # Receive the header
    data_header = int.from_bytes(data_header, 'big')  # Convert header to integer

    data_length = int.from_bytes(socket.recv(4), 'big')  # Receive the length of the data
    
    bytes_received = 0
    data = b""

    while bytes_received < data_length:
        chunk = socket.recv(min(buffer_size, data_length - bytes_received))
        if not chunk:
            break
        data += chunk
        bytes_received += len(chunk)
    
    if data_header == HEADER_TERMINATE:
        return None
    
    return data

def measure_timelag(socket_rx: socket.socket, socket_tx: socket.socket, node_type: str) -> float:
    """
    Measure the time lag between sending and receiving data over a socket.
    ping-pong and measure the RTT.
    client time + lag = server time
    server time - lag = client time

    Args:
        socket_rx (socket.socket): The socket object to receive data from.
        socket_tx (socket.socket): The socket object to send data to.
        node_type (str): The type of connection ('server' or 'client').

    Returns:
        float: The time lag in seconds.
    """
    num_repeats = 10
    lags = []

    if node_type == "client":
        for _ in range(num_repeats):
            timestamp_client_send = time.time()
            timestamp_client_send_data = str(timestamp_client_send).encode('utf-8')
            transmit_data(socket_tx, timestamp_client_send_data)
            data = receive_data(socket_rx)
            timestamp_server = float(data.decode('utf-8'))
            timestamp_client_recv = time.time()

            client_mid = (timestamp_client_send + timestamp_client_recv) / 2
            
            lag = timestamp_server - client_mid
            lags.append(lag)

        lag_avg = sum(lags) / len(lags)
        transmit_data(socket_tx, str(lag_avg).encode('utf-8'))

    elif node_type == "server":
        for _ in range(num_repeats):
            data = receive_data(socket_rx)
            timestamp_server = str(time.time()).encode('utf-8')
            transmit_data(socket_tx, timestamp_server)

        data = receive_data(socket_rx)
        lag_avg = float(data.decode('utf-8'))

    else:
        raise ValueError("Invalid type. Use 'server' or 'client'.")

    return lag_avg


# Example usage
if __name__ == "__main__":
    import sys
    import random

    # get the first system arguments
    proc_type = sys.argv[1] if len(sys.argv) > 1 else "server"

    socket_rx, socket_tx = connect_dual_tcp("localhost", (9000, 9001), node_type=proc_type)
    
    # random test message
    test_message = "Hello, this is a test message. "
    test_message += str(random.randint(0, 10000000))

    if proc_type == "server":
        transmit_data(socket_tx, test_message.encode())
        print(f"Server sent: {test_message}")
        received = receive_data(socket_rx)
        print(f"Server received: {received.decode()}")

        lag_avg = measure_timelag(socket_rx, socket_tx, proc_type)
        print(f"Average lag: {lag_avg:.6f} seconds")

    elif proc_type == "client":
        received = receive_data(socket_rx)
        print(f"Client received: {received.decode()}")
        transmit_data(socket_tx, test_message.encode())
        print(f"Client sent: {test_message}")

        lag_avg = measure_timelag(socket_rx, socket_tx, proc_type)
        print(f"Average lag: {lag_avg:.6f} seconds")
    
    socket_rx.close()
    socket_tx.close()
    print("Sockets closed.")

