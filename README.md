### MedRank-IQA
This work explores the application of learning-to-rank methodologies to the problem of image quality assessment in computed tomography (CT).

### Environment creation 

### TODO: 
- pretraining
    - stesso random crop per x e x^
    - diverso random crop per x e x^
    - integra caso x^1 vs x^2 invece di solo x vs x^ (con x^1 e x^2 stesso artefatto e livello a meno ogni volta di un parametro diverso per cui possa avere senso un ranking tra le due)

- finetuning
    - early stopping (usiamo come validation set le 1000-912 immagini non usate per test con dati bilanciati tra gli artefatti)
    - test con backbone frozen + train solo della head (**situazione attuale**)
    - test con due optimizer, uno per head e uno per backbone (lr più basso)
    - test sbloccando backbone dopo un tot di epoche ...

- for inference (test del modello finetuned) 30 random patches and compute median (see Rank-IQA paper + https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/main.py)

- artifact simulation from projections
    - modify the code for starting from projections ... 
    - streak - ok (try also limited angles)
    - noise - to check
    - from full projections I can recostruct different ranges of the helix acquisition...