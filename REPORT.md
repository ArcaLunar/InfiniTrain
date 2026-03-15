## Experiment

```bash
CUDA_VISIBLE_DEVICES=5,6,7 ./gpt2 --input_bin ../../data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin --llmc_filepath ../../data/llmc/gpt2/gpt2_124M.bin --checkpoint_dir ../ckpt2/gpt2-noresume/ --num_iteration 100 --save_steps 20 --save_optimizer_state true --max_checkpoint_keep 10
```

```bash
CUDA_VISIBLE_DEVICES=5,6,7 ./gpt2 --input_bin ../../data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin --llmc_filepath ../../data/llmc/gpt2/gpt2_124M.bin --checkpoint_dir ../ckpt2/gpt2-resumefrom40/ --num_iteration 100 --save_steps 20 --save_optimizer_state true --max_checkpoint_keep 10 --resume_from ../ckpt2/gpt2-noresume/checkpoint_step_000040/ > ../ckpt2/gpt2-resumefrom40/gpt2-resume.log 2>&1
```

运行 compare_loss.py，对于 llama3 模型，由于从 step 40 恢复训练，所以 step 1~40 数据缺失，而其余 60 步的 loss 在 FP32, BF16 下均吻合

```text
  Summary: 60/100 steps matched

==================================================
Overall Summary:
  fp32:    0/1 test cases passed (threshold: 1e-05)
  bfloat16: 0/0 test cases passed (threshold: 1e-02)
  Total:   0/1 test cases passed
==================================================

==================================================
Overall Summary:
  fp32:    0/0 test cases passed (threshold: 1e-05)
  bfloat16: 0/1 test cases passed (threshold: 1e-02)
  Total:   0/1 test cases passed
==================================================
```

对于 GPT2，模型保存的逻辑有误：训练中 lm_head 与 wte 并非真共享，而 LLMC 存取又按“共享”假设处理，resume 后 lm_head 很容易和 noresume 不一致。解决方法是把训练用 checkpoint 从 LLMC 回调路径切到原生 StateDict 二进制路径，并在加载后显式重建权重绑定语义 (`example/gpt2/main.cc`)．经过修复后，也可以通过．
