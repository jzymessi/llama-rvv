CC=/workspace/riscv-gnu-toolchain/install/bin/riscv64-unknown-linux-gnu-gcc
RUN=/workspace/riscv-gnu-toolchain/install/bin/qemu-riscv64
spike=/workspace/riscv-isa-sim/install/bin/spike
pk=/workspace/riscv-pk/install/riscv64-unknown-elf/bin/pk

# ${CC} -static -g  add.s hello.c  -o hello
# ${spike} ${pk} softmax 

${CC} -g -o run_rvv run_rvv.c -march=rv64gcv -lm -static 
${RUN} -cpu rv64,v=true,vlen=128,vext_spec=v1.0 run_rvv ./stories15M.bin
