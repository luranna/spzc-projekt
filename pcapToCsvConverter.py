from scapy.all import rdpcap, IP, TCP, UDP, ICMP

packets = rdpcap('/home/ubuntu/Downloads/ttl-dataset/text.pcap')

from collections import defaultdict
import time

def extract_basic_features(packet):
    features = {}
    if IP in packet:
        features['timestamp'] = packet.time  # Timestamp of the packet
        features['src_ip'] = packet[IP].src
        features['dst_ip'] = packet[IP].dst
        features['ip_version'] = packet[IP].version
        features['header_length'] = packet[IP].ihl
        features['tos'] = packet[IP].tos
        features['total_length'] = packet[IP].len
        features['identification'] = packet[IP].id
        features['ttl'] = packet[IP].ttl
        features['protocol'] = packet[IP].proto
        features['checksum'] = packet[IP].chksum
        features['ip_flags'] = packet[IP].flags
        features['fragment_offset'] = packet[IP].frag
        if TCP in packet:
            features['src_port'] = packet[TCP].sport
            features['dst_port'] = packet[TCP].dport
            features['seq'] = packet[TCP].seq
            features['ack'] = packet[TCP].ack
            features['data_offset'] = packet[TCP].dataofs
            features['reserved'] = packet[TCP].reserved
            features['tcp_flags'] = packet[TCP].flags
            features['window'] = packet[TCP].window
            features['checksum'] = packet[TCP].chksum
            features['urgent_pointer'] = packet[TCP].urgptr
            features['src_bytes'] = len(packet[TCP].payload)
            features['dst_bytes'] = 0  # Needs stateful analysis to track response
        elif UDP in packet:
            features['src_port'] = packet[UDP].sport
            features['dst_port'] = packet[UDP].dport
            features['length'] = packet[UDP].len
            features['checksum'] = packet[UDP].chksum
            features['src_bytes'] = len(packet[UDP].payload)
            features['dst_bytes'] = 0  # Needs stateful analysis to track response
        elif ICMP in packet:
            features['type'] = packet[ICMP].type
            features['code'] = packet[ICMP].code
            features['checksum'] = packet[ICMP].chksum
            features['icmp_id'] = packet[ICMP].id
            features['icmp_seq'] = packet[ICMP].seq
    return features


def extract_content_features(packet):
    features = {}
    # This requires deep packet inspection for application-layer data
    # Example: HTTP headers, FTP commands, etc.
    return features

def extract_time_based_features(packets):
    features = defaultdict(lambda: defaultdict(int))
    start_time = packets[0].time
    two_second_window = []
    for packet in packets:
        current_time = packet.time
        if current_time - start_time > 2:
            two_second_window.pop(0)
        two_second_window.append(packet)
        # Compute features over two_second_window
        # features['count'], 'srv_count', etc.
    return features

def extract_host_based_features(packets):
    features = defaultdict(lambda: defaultdict(int))
    # Similar to time-based but over last 100 connections
    # features['dst_host_count'], 'dst_host_srv_count', etc.
    return features

all_features = []
for packet in packets:
    features = {}
    features.update(extract_basic_features(packet))
    features.update(extract_content_features(packet))
    # Append time-based and host-based features as needed
    all_features.append(features)

# Post-process features to add time-based and host-based features
time_based_features = extract_time_based_features(packets)
host_based_features = extract_host_based_features(packets)

# Combine all features
for i, packet_features in enumerate(all_features):
    packet_features.update(time_based_features[i])
    packet_features.update(host_based_features[i])

import pandas as pd

df = pd.DataFrame(all_features)
df.to_csv('/home/ubuntu/Desktop/spzc/data/ttl-benign-data.csv', index=False)
