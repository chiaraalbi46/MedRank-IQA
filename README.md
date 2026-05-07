### MedRank-IQA
This work explores the application of learning-to-rank methodologies to the problem of image quality assessment in computed tomography (CT).

### Environment creation 

### Files content

- pretraining_rank_iqa.py
- pretraining_rank_iqa_balanced_batches_v1.py
- pretraining_networks.py
- finetuning_rank_iqa.py

### TODO: 

| Model | Mode                                      | N=1000 | N=300 | N=60 | N=12 |
| ----- | -------------------------------------     | ------ | ----- | ---- | ---- |
| VGG16 | from scratch                              |        |       |      |      |
| VGG16 | init imagenet                             |        |       |      |      |
| VGG16 | pretraining ranking CT + finetuning       |        |       |      |      |
| VGG16 | pretraining ranking NAT + finetuning      |        |       |      |      |
| RESENET18 | from scratch                          |        |       |      |      |
| RESENET18 | init imagenet                         |        |       |      |      |
| RESENET18 | pretraining ranking CT + finetuning   |        |       |      |      |
| SQUEEZENET1.1 | from scratch                      |        |       |      |      |
| SQUEEZENET1 | init imagenet                       |        |       |      |      |
| SQUEEZENET1 | pretraining ranking CT + finetuning |        |       |      |      |

*eventualmente init imagenet + pretraining ranking CT + finetuning
**linear probe oltre a finetuning 

### Lista modelli da cui estrarre le features

Modelli pretrained 
- '/data/staff/albisani/MedRank-IQA/pretrained_models_NEW/full_syn_real_balanced_v1_or_per_0.5.pth'  # VGG16, pretraining ranking CT 
- '/data/staff/albisani/MedRank-IQA/pretrained_models_NEW/full_syn_real_balanced_v1_or_per_0.5_squeezenet.pth'  # SQUEEZENET1.1, pretraining ranking CT
- '/data/staff/albisani/MedRank-IQA/Pytorch_TestRankIQA_pretrained/Rank_tid2013.caffemodel.pt'  # VGG16 pretraining ranking NAT

Modelli from scratch/init imagenet (posso estrarre solo features TEST set. Non posso fare linear/non linear probing qui)
- '/data/staff/albisani/MedRank-IQA/RESULTS_VGG/fs_lr_backbone/best_model.pth'  # VGG16, from scratch, 1000
- '/data/staff/albisani/MedRank-IQA/RESULTS_VGG/fs_imagenet_lr_backbone/best_model.pth'  # VGG16, init imagenet, 1000
- '/data/staff/albisani/MedRank-IQA/RESULTS_VGG/fs_lr_backbone_12/best_model.pth'  # VGG16, from scratch, 12
- '/data/staff/albisani/MedRank-IQA/RESULTS_VGG/fs_imagenet_lr_backbone_12/best_model.pth'  # VGG16, init imagenet, 12
