* Iteration 500 vs 1000
  
* Activation (Sigmoid vs. ReLU)
  I've observed the sigmoid activation converage more gradually while ReLU make the convergence little bit sparsly. And since ReLU didn't help to improve accuracy, I used the sigmoid as a default.

* Number of Linear

* Randomness of LeNet 
  Initially, model wasn't deterministic and produced wide ranges of accuracies given a configuration. So I had to walk around by seeding the random module.

  ```python
    import torch
    # ...
    torch.manual_seed(1)
  ```