**Visualize**
```
python -m grpo_smolvla.visualize --suite libero_10 --task_id 5 --n_rollouts 3 --output vis_libero_10_task5.png
```

**Evaluate**
```
python -m grpo_smolvla.evaluate --checkpoint lerobot/smolvla_libero --suites libero_10 --n_episodes 20
```

Last time (2026-05-06 08:00)
```
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  [libero_10] task  0: 20.00%  put both the alphabet soup and the tomato sauce in the baske
  [libero_10] task  1: 60.00%  put both the cream cheese box and the butter in the basket
  [libero_10] task  2: 85.00%  turn on the stove and put the moka pot on it
  [libero_10] task  3: 95.00%  put the black bowl in the bottom drawer of the cabinet and c
  [libero_10] task  4: 25.00%  put the white mug on the left plate and put the yellow and w
  [libero_10] task  5: 70.00%  pick up the book and place it in the back compartment of the
  [libero_10] task  6: 30.00%  put the white mug on the plate and put the chocolate pudding
  [libero_10] task  7: 40.00%  put both the alphabet soup and the cream cheese box in the b
  [libero_10] task  8: 35.00%  put both moka pots on the stove
  [libero_10] task  9: 55.00%  put the yellow and white mug in the microwave and close it
  [libero_10] Mean success rate: 51.50%
```


```
 python -m grpo_smolvla.train_grpo --config configs/grpo_config.yaml
```