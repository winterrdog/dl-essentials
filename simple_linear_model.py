import tensorflow as tf


class LinearModel(tf.Module):
    """base model for my small model."""

    def __init__(self):
        self.weight = tf.Variable(1.0)
        self.bias = tf.Variable(1.0)

    def __call__(self, X):
        return (self.weight * X) + self.bias


# loss function
def loss_mse(X, Y, model):
    Y_hat = model(X)
    diff = Y_hat - Y

    return tf.reduce_mean(diff**2)


# training part
def train_step(X, Y, model, optimizer):

    # for automatic differentiation during training
    def compute_gradient(X, Y, model):
        with tf.GradientTape() as tape:
            loss = loss_mse(X, Y, model)

        return tape.gradient(loss, [model.weight, model.bias])

    gradients = compute_gradient(X, Y, model)
    optimizer.apply_gradients(zip(gradients, [model.weight, model.bias]))


def main():
    # create a model and optimizer
    model = LinearModel()
    optimizer = tf.optimizers.SGD(learning_rate=0.01)

    # dummy data
    X, Y = (
        tf.constant([1.0, 2.0, 3.0, 4.0]),
        tf.constant([2.0, 3.0, 4.0, 5.0]),
    )

    # training loop
    for epoch in range(100):
        train_step(X, Y, model, optimizer)
        curr_loss = loss_mse(X, Y, model)
        print(f"Epoch {epoch}: Loss = {curr_loss}")

    print(
        f"Trained weights: weight = {model.weight.numpy()}, bias = {model.bias.numpy()}"
    )


if __name__ == "__main__":
    main()
