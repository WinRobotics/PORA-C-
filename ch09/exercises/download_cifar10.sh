#!/usr/bin/env bash
# Download the binary version of CIFAR-10 (~170 MB) into ch09/exercises/data/
# (the counterpart of `torchvision.datasets.CIFAR10(..., download=True)`).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
if [ -d data/cifar-10-batches-bin ]; then
  echo "CIFAR-10 already present in $(pwd)/data/cifar-10-batches-bin"
  exit 0
fi
curl -L -o data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf data/cifar-10-binary.tar.gz -C data
rm data/cifar-10-binary.tar.gz
echo "CIFAR-10 extracted to $(pwd)/data/cifar-10-batches-bin"
