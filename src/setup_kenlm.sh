#!/bin/bash

sudo apt install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev
mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
