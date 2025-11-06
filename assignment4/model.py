''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from typing import Protocol, Tuple

import numpy as np

# set the numpy random seed so our randomness is reproducible
np.random.seed(67)


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the Perceptron classes further down.
class Model(Protocol):
    def get_hyperparams(self) -> dict:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...


class MajorityBaseline(Model):
    def __init__(self):
        print('MAJORITY BASELINE INIT')
        self.majority_label = None
    
    def get_hyperparams(self) -> dict:
        return {}
    
    def train(self, x: np.ndarray, y: np.ndarray):
        values, counts = np.unique(y, return_counts=True)
        self.majority_label = values[np.argmax(counts)]
        
    def predict(self, x: np.ndarray) -> list:
        if self.majority_label is None:
            raise ValueError("Model has not been trained yet.")
        return [self.majority_label] * len(x)


class Perceptron(Model):
    def __init__(self, num_features: int, lr: float, decay_lr: bool = False, mu: float = 0):
        '''
        Initialize a new Perceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate (eta). This is also the initial learning rate if decay_lr=True
            decay_lr (bool): whether or not to decay the initial learning rate lr
            mu (float): the margin (mu) that determines the threshold for a mistake. Defaults to 0
        '''
        self.num_features = num_features #I added this, don't know why it wasn't included with the other three
        self.lr = lr
        self.decay_lr = decay_lr
        self.mu = mu
        # I added everything below here
        self.w = np.random.uniform(low=-0.001, high=0.001, size=num_features)
        self.b = 0.0
        self.t = 0
        self.num_updates = 0


    def get_hyperparams(self) -> dict:
        return {'lr': self.lr, 'decay_lr': self.decay_lr, 'mu': self.mu}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        n_samples = x.shape[0]

        # run the update rule for a number of epochs
        for _ in range(epochs):
            x, y = shuffle_data(x, y)
            lr = self.lr

            if self.decay_lr:
                lr = self.lr / (1 + self.t)

            for i in range(n_samples):
                x_i = x[i]
                y_i = y[i]

                activation = np.dot(self.w, x_i) + self.b

                if y_i * activation <= self.mu:
                    self.w = self.w + lr * y_i * x_i
                    self.b = self.b + lr * y_i
                    self.num_updates += 1

            self.t += 1
        #uncomment line below to see how many updates a training session takes
        print(f'Number of updates: {self.num_updates}')
    

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        activations = np.dot(x, self.w) + self.b
        preds = np.where(activations >= 0, 1, -1)
        return preds.tolist()
    

class AveragedPerceptron(Model):
    def __init__(self, num_features: int, lr: float):
        '''
        Initialize a new AveragedPerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate eta
        '''     

        self.lr = lr
        self.num_features = num_features

        self.w = np.zeros(num_features)
        self.b = 0.0

        self.w_sum = np.zeros(num_features)
        self.b_sum = 0.0

        self.counter = 0
        self.num_updates = 0


    def get_hyperparams(self) -> dict:
        return {'lr': self.lr}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        '''
        Train using the Averaged Perceptron algorithm.
        '''
        n_samples = x.shape[0]

        for _ in range(epochs):
            x, y = shuffle_data(x, y)

            for i in range(n_samples):
                self.counter += 1
                x_i = x[i]
                y_i = y[i]

                activation = np.dot(self.w, x_i) + self.b

                # basic perceptron update
                if y_i * activation <= 0:
                    self.w += self.lr * y_i * x_i
                    self.b += self.lr * y_i
                    self.num_updates += 1

                # cumulative sum of weights and bias for averaging
                self.w_sum += self.w
                self.b_sum += self.b

        # compute averages
        self.w_avg = self.w_sum / self.counter
        self.b_avg = self.b_sum / self.counter
        print(f'Number of updates: {self.num_updates}')
    

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        activations = np.dot(x, self.w_avg) + self.b_avg
        preds = np.where(activations >= 0, 1, -1)
        return preds.tolist()
    

class AggressivePerceptron(Model):
    def __init__(self, num_features: int, mu: float):
        '''
        Initialize a new AggressivePerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            mu (float): the hyperparameter mu
        '''     

        self.mu = mu
        
        # YOUR CODE HERE


    def get_hyperparams(self) -> dict:
        return {'mu': self.mu}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
    

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        return []


# DON'T MODIFY THE FUNCTIONS BELOW!
PERCEPTRON_VARIANTS = ['simple', 'decay', 'margin', 'averaged', 'aggressive']
MODEL_OPTIONS = ['majority_baseline'] + PERCEPTRON_VARIANTS
def init_perceptron(variant: str, num_features: int, lr: float, mu: float) -> Model:
    '''
    This is a helper function to help you initialize the correct variant of the Perceptron

    Args:
        variant (str): which variant of the perceptron to use. See PERCEPTRON_VARIANTS above for options
        num_features (int): the number of features (i.e. dimensions) the model will have
        lr (float): the learning rate hyperparameter eta. Same as initial learning rate for decay setting
        mu (float): the margin hyperparamter mu. Ignored for variants "simple", "decay", and "averaged"

    Returns
        (Model): the initialized perceptron model
    '''
    
    assert variant in PERCEPTRON_VARIANTS, f'{variant=} must be one of {PERCEPTRON_VARIANTS}'

    if variant == 'simple':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=False)
    elif variant == 'decay':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True)
    elif variant == 'margin':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True, mu=mu)
    elif variant == 'averaged':
        return AveragedPerceptron(num_features=num_features, lr=lr)
    elif variant == 'aggressive':
        return AggressivePerceptron(num_features=num_features, mu=mu)


def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''

    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]
