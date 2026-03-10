# DNNKit Experimental Report

## Overview
This report summarizes benchmark experiments run with DNNKit.

## Best Run
- Dataset: mnist
- Epochs: 3
- Learning rate: 0.001
- Batch size: 64
- Final loss: 0.1118
- Final test accuracy: 0.9697
- Device: cpu
- Run directory: outputs/mnist_20260310_033745

## Leaderboard
| dataset   |   epochs |    lr |   batch_size |   final_loss |   final_test_accuracy | device   | run_dir                       |
|:----------|---------:|------:|-------------:|-------------:|----------------------:|:---------|:------------------------------|
| mnist     |        3 | 0.001 |           64 |     0.11183  |                0.9697 | cpu      | outputs/mnist_20260310_033745 |
| mnist     |        3 | 0.001 |           64 |     0.111004 |                0.9697 | cpu      | outputs/mnist_20260310_034234 |

## Summary
The best-performing experiment achieved a final test accuracy of
0.9697 on the mnist benchmark.
This demonstrates that DNNKit supports reproducible training,
benchmarking, and experiment comparison workflows.
