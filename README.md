# llama-rvv

之前看到llama2.c的项目<https://github.com/karpathy/llama2.c>,以及最近了解了一些RISCV，就想试着用RVV(RISC_V Vector)重写llama2模型中的算子，并在spike模拟器上运行。

## build

### riscv-isa-sim
<https://github.com/riscv-software-src/riscv-isa-sim>

### riscv-gnu-toolchain
<https://github.com/riscv-collab/riscv-gnu-toolchain>

### riscv-pk
<https://github.com/riscv-software-src/riscv-pk>

## Run

```
bash run.sh
```

## Note
目前只重写了llama2中的softmax和rmsnorm这两个算子。

## Ref
<https://github.com/riscv-non-isa/rvv-intrinsic-doc>

<https://github.com/Tencent/ncnn/tree/master/src/layer/riscv>

<https://github.com/karpathy/llama2.c>