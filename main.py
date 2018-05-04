import torch
import torch.nn as nn
import torch.optim as optim
from linear_regression import LinearRegression
from plotter import Plotter


if __name__ == '__main__':

    # hyper parameters
    input_size = 1
    output_size = 1
    epochs = 50
    lr = 0.001

    # dummy data
    x_train = torch.arange(20).unsqueeze(-1)
    delta = torch.randn(20, 1)
    y_train = torch.add(x_train, delta)

    # instantiate model
    model = LinearRegression(input_size, output_size)

    # instantiate loss criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    x_train_np = x_train.detach().numpy()
    y_train_np = y_train.detach().numpy()

    ims = []
    for epoch in range(epochs):

        # empty the gradients of the optimizer
        optimizer.zero_grad()

        # forward pass
        output = model(x_train)

        # compute loss
        loss = criterion(output, y_train)

        # compute gradients using backpropagation
        loss.backward()

        # update parameters using optimizer.step()
        optimizer.step()

        # print loss
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, loss.item()))
        output_np = output.detach().numpy()
        ims.append([epoch, x_train_np, output_np])

    # plot the results
    plotter = Plotter(x_train_np, y_train_np, ims)
    plotter.plot()
