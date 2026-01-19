### MedRank-IQA
This work explores the application of learning-to-rank methodologies to the problem of image quality assessment in computed tomography (CT).

### Environment creation 

### TODO: 
- create file for matching images path and names with artifacts - to do 
- work on sampling considering artifact types
- keep common part in same file (avoid re-writing some stuff in different points)
- work on preprocessing - avoid resize, use central crop and then random patch during training 
    - x, x^ and then select a random patch from x and from x^ (not in the same position)
- for inference 30 random patches and compute median (see Rank-IQA paper + https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/main.py)

- artifact simulation for pretraining (phase 1) [consider working on a separate branch ...]
- modify the code for starting from projections ... 
- streak - ok (try also limited angles)
- noise - to check
