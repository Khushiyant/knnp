


# kNN-P Classifier (Under Development)

[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)


## Overview

![Screenshot 2023-11-01 at 11 45 08 PM](https://github.com/Khushiyant/knnp/assets/69671407/fdf509df-7a65-41c6-885a-5032d04f7bbb)

This repository contains an implementation of the kNN-P classifier, an enhanced version of the k-nearest neighbors algorithm utilizing membrane computing. kNN-P is designed for parallel and distributed computing, which can improve the performance of the original k-nearest neighbors algorithm for classification tasks.

**Please note that this project is currently under development.**

## Features

- Implementation of the kNN-P classifier.
- Designed for parallel and distributed computing.
- Improved performance for classification tasks.

## Installation

You can install this package using pip:

```bash
pip install kNNp
```

## Usage

```python
from kNNp.p_systems import kNN_P

# Create an instance of kNN-P
knn_p = kNN_P(n=100, d=2, q=10, m=5, k=3, maxstep=100)

# Load your training data (features) and class labels
training_data = ...
class_labels = ...

# Train the classifier
knn_p.fit(training_data, class_labels)

# Load your test data
test_data = ...

# Make predictions
predictions = knn_p.predict(test_data)

# Evaluate the predictions and calculate classification metrics
...
```

## Output
```bash

# training_data = np.random.random_sample(size=(100, 9))
# class_labels = np.random.randint(0, 2, size=(100, 1))

# for i, (neighbors_indices, _) in enumerate(predictions):
#     print("Test sample {} is classified as class {}".format(i, y[neighbors_indices[0]]) )

Test sample 0 is classified as class [0]
Test sample 1 is classified as class [1]
Test sample 2 is classified as class [1]
Test sample 3 is classified as class [1]
Test sample 4 is classified as class [1]
Test sample 5 is classified as class [1]
Test sample 6 is classified as class [1]
Test sample 7 is classified as class [1]
Test sample 8 is classified as class [1]
Test sample 9 is classified as class [1]
```

## Contributing

Contributions to this project are welcome. Please follow these guidelines for contributing:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push your branch to your fork.
5. Create a pull request with a clear description of your changes.

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

If you have questions or need further assistance, please feel free to reach out to [Khushiyant](mailto:khushiyant2002@gmail.com).

---

**Please note that this project is under development. Use it with caution, and contributions are encouraged.**

## References

Hu, Juan, et al. “KNN-P: A KNN Classifier Optimized by P Systems.” Theoretical Computer Science, vol. 817, 1 May 2020, pp. 55–65, www.sciencedirect.com/science/article/abs/pii/S0304397520300098, https://doi.org/10.1016/j.tcs.2020.01.001. Accessed 30 Oct. 2023.
