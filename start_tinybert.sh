#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
cd TinyBERT
python3 realtime_network_monitor.py --interactive
