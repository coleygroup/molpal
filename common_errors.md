# MolPAL: Molecular Pool-based Active Learning
# Common Errors and Hor to Deal with Them

## Memory Errors 
### During multi-objective acquisition (usually EI, sometimes PI)
The multi-objective EI and PI (the latter to a lesser extent) calculate a LOT of values. Generally it scales with the number of points on the PF^2 and the number of evaluated points. This means that as the number of objectives increases (or the PF for another reason has many points on it), your EI calculations will explode. To alleviate this issue, we use ray to parallelize such calculations. If you are running into memory errors here, we recommend changing the default batch size in `chunked_ehvi` or `chunked_pi` in `metrics.py`. In the current code the default batch size here is always used. Add an option for it in `args.py` if you want to!

### MPN Inference 
The MPN model class does its own batching during inference (this maybe should be changed in the long run). In the MPN predict code, `ray.put(model)` is called. It is bad for memory to call this iteratively (see [ray documentation](https://docs.ray.io/en/latest/ray-core/examples/batch_prediction.html#batch-prediction)). So, you should not be doing any batching in the base model class when using an MPN. (This should already happen in the base multi-objective model class but could be useful for debugging.) NOTE: this HELPS the problem but doesn't fully solve it because ray::IDLEs still accumulate. debugging in process now for this issue. Lowering the `--ncpu` argument might help. 

## Non-errors 
### GPU available but not u
## Other 
### Tmux being annoying 
Super random and not related to MolPAL, but sometimes code that runs in the normal terminal does not run in tmux for me. This came down to tmux not using the python in my conda environment even though it was activated. At this moment I forget the exact solution, but essentially you have to activate the environment at a different time. i.e. Do not activate the environment, call `tmux a -t my_tmux` and THEN activate your desired environment. 

### ADFR Suite 
You have to manually install this, don't use conda like I did 
