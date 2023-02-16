## batch.py的用法
用来批量跑rate和convergence的实验
eg.
```bash
python3.9 batch.py --task rate --dgp_p 1 --dgp_r 4
```

## run.py的用法
1. 单独使用run.py时尽量只选择 --task debug （以免混淆实验记录）
2. 用于调试

```
python3.9 run.py --n_rep 1 --T 1000 --T0 10 --s 10 --A_init GD --print_log
```