* Iteration 500 vs 1000
  The iteration was one of the parameter in sweep but found that the model usually converges enough to test accruacy around 500 if the configuration works.

* Activation (Sigmoid vs. ReLU)
  I've observed the sigmoid activation converage more gradually while ReLU make the convergence little bit sparsly. And since ReLU didn't help to improve accuracy, I used the sigmoid as a default.

* Number of Down-sampling Layers
  Similar to LeNet-5 architecture, I started with three layers to produce single output node:

  1) C5: Layer 144 -> 120
  2) F6: Layer 120 -> 84
  3) Hidden layer 84 -> 20
  4) Output Layer 20 -> 1

* Randomness of LeNet 
  Initially, model wasn't deterministic and produced wide ranges of accuracies given a configuration. So I had to walk around by seeding the random module.

  ```python
    import torch
    # ...
    torch.manual_seed(1)
  ```

  * Data Augmentation
    I increase the volumn of training set by augmenting images with three methods; rotation, noise, flipping.

    * Using training data including rotated images improved accuracy upto 95.29%. (Left only)
    * Traning samples with random noise didn't pass the bar, 92.5%.
    * 

* SoftMax
* Effect on current accuracy in current architecture.