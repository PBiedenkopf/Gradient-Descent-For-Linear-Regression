# Gradient descent for linear regression
This implementation of linear regression uses gradien descent for minimization of the cost function. Optionally polynomial features can be added with a regularized cost function.

## Datasets
The included datasets in the examples are open datasets taken from [kaggle](https://www.kaggle.com/). 
To use them in linear regression some parameters of the datasets are modified from string to belonging numerous values.

### [Fish market dataset](https://www.kaggle.com/aungpyaeap/fish-market)
Target is to predict the weight of a fish with known body measures and species. Published under
GPL 2 license.

Parameters:
- Species (1-Bream, 2-Roach, 3-Whitefish, 4-Parkki, 5-Perch, 6-Pike, 7-Smelt)
- Weight
- Length 1
- Length 2
- Length 3
- Height
- Width

## License
MIT License

Copyright (c) 2020 Philipp Biedenkopf
