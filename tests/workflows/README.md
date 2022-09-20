# Run the test

By default the test will run with 500 train datasamples and 200 test samples. If you wish to run it with more samples (70k available) set it in `values.yaml`
```
mnist_workflow:
  train_samples: <nb train samples>
  test_samples: <nb test samples>
```

## run locally with docker or subprocess

`pytest tests -v --durations=0 -m "workflows" --subprocess`
