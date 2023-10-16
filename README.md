# 


```
$ petry run pytest -s -v tests
```

# Known Issues

- gmm.py
  Large coavariance values causes np.linalg.LinAlgError about singularity.
  Theoretically, this computational problem can be escaped by using log-scale computation.

- Inital varialbes of models are not appropreate.

- Evaluation by estimated states of HMMs.