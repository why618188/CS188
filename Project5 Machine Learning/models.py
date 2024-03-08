import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        value = nn.as_scalar(self.run(x))
        if value >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            flag = True
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                label = nn.as_scalar(y)
                if label != prediction:
                    self.w.update(x, label)
                    flag = False
                    continue
            if flag:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        """
        Initialize your model parameters here
        """
        "*** YOUR CODE HERE ***"
        self.batchSize = 200
        self.hiddenLayerSize = 512
        self.alpha = 0.03
        self.W1 = nn.Parameter(1, self.hiddenLayerSize)
        self.b1 = nn.Parameter(1, self.hiddenLayerSize)
        self.W2 = nn.Parameter(self.hiddenLayerSize, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        return nn.AddBias(nn.Linear(h1, self.W2), self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(y, predicted_y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batchSize):
            loss = self.get_loss(x, y)
            if nn.as_scalar(loss) <= 0.01:
                break
            grad_W1, grad_W2, grad_b1, grad_b2 = nn.gradients(loss, [self.W1, self.W2, self.b1, self.b2])
            self.W1.update(grad_W1, -1.0 * self.alpha)
            self.W2.update(grad_W2, -1.0 * self.alpha)
            self.b1.update(grad_b1, -1.0 * self.alpha)
            self.b2.update(grad_b2, -1.0 * self.alpha)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        """
        Initialize your model parameters here
        """
        "*** YOUR CODE HERE ***"
        self.batchSize = 100
        self.hiddenLayerSize = 256
        self.alpha = 0.5
        self.inputSize = 784
        self.outputSize = 10
        self.W1 = nn.Parameter(self.inputSize, self.hiddenLayerSize)
        self.b1 = nn.Parameter(1, self.hiddenLayerSize)
        self.W2 = nn.Parameter(self.hiddenLayerSize, self.outputSize)
        self.b2 = nn.Parameter(1, self.outputSize)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        return nn.AddBias(nn.Linear(h1, self.W2), self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        step = 0
        for x, y in dataset.iterate_forever(self.batchSize):
            step += 1
            loss = self.get_loss(x, y)
            grad_W1, grad_W2, grad_b1, grad_b2 = nn.gradients(loss, [self.W1, self.W2, self.b1, self.b2])
            self.W1.update(grad_W1, -1.0 * self.alpha)
            self.W2.update(grad_W2, -1.0 * self.alpha)
            self.b1.update(grad_b1, -1.0 * self.alpha)
            self.b2.update(grad_b2, -1.0 * self.alpha)
            if step % 50 != 0:
                continue
            if dataset.get_validation_accuracy() >= 0.98:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 100
        self.alpha = 0.05
        self.hiddenSize1 = 512
        self.hiddenSize2 = 256
        self.W1 = nn.Parameter(self.num_chars, self.hiddenSize1)
        self.W_hidden = nn.Parameter(5, self.hiddenSize1)
        self.b1 = nn.Parameter(1, self.hiddenSize1)
        self.W2 = nn.Parameter(self.hiddenSize1, self.hiddenSize2)
        self.b2 = nn.Parameter(1, self.hiddenSize2)
        self.W3 = nn.Parameter(self.hiddenSize2, 5)
        self.b3 = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for i in range(len(xs)):
            if i == 0:
                z = nn.Linear(xs[i], self.W1)
            else:
                z = nn.Add(nn.Linear(xs[i], self.W1), nn.Linear(H, self.W_hidden))
            h1 = nn.ReLU(nn.AddBias(z, self.b1))
            h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.W2), self.b2))
            H = nn.AddBias(nn.Linear(h2, self.W3), self.b3)
        return H

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(xs)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        step = 0
        for x, y in dataset.iterate_forever(self.batchSize):
            step += 1
            loss = self.get_loss(x, y)
            grad_W1, grad_W_hidden, grad_W2, grad_b1, grad_b2 = nn.gradients(loss, [self.W1, self.W_hidden, self.W2, self.b1, self.b2])
            self.W1.update(grad_W1, -1.0 * self.alpha)
            self.W2.update(grad_W2, -1.0 * self.alpha)
            self.b1.update(grad_b1, -1.0 * self.alpha)
            self.b2.update(grad_b2, -1.0 * self.alpha)
            self.W_hidden.update(grad_W_hidden, -1.0 * self.alpha)
            if step % 50 != 0:
                continue
            if dataset.get_validation_accuracy() >= 0.85:
                break
