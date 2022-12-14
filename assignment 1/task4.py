import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import utils
import dataloaders
import torchvision
from trainer import Trainer
torch.random.manual_seed(0)
np.random.seed(0)


# Load the dataset and print some stats
batch_size = 64

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform)
example_images, _ = next(iter(dataloader_train))
print(f"The tensor containing the images has shape: {example_images.shape} (batch size, number of color channels, height, width)",
      f"The maximum value in the image is {example_images.max()}, minimum: {example_images.min()}", sep="\n\t")


def create_model():
    """
        Initializes the mode. Edit the code below if you would like to change the model.
    """
    model = nn.Sequential(
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        nn.Linear(28*28*1, 10)
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model


model = create_model()


# Test if the model is able to do a single forward pass
example_images = utils.to_cuda(example_images)
output = model(example_images)
print("Output shape:", output.shape)
expected_shape = (batch_size, 10)  # 10 since mnist has 10 different classes
assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"


# Hyperparameters
learning_rate = .0192
num_epochs = 5


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)


trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_loss_dict, test_loss_dict = trainer.train(num_epochs)

torch.save(model.state_dict(), "saved_model.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_test, model, loss_function)
print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")

# Task4 (a) normalized version:
torch.random.manual_seed(0)
np.random.seed(0)

image_transform_normalized = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(0.5,0.5)
])
dataloader_train_normalized, dataloader_test_normalized = dataloaders.load_dataset(
    batch_size, image_transform_normalized)
example_images_normalized, _normalized = next(iter(dataloader_train_normalized))

model_normalized = create_model()

# Define optimizer (Stochastic Gradient Descent)
optimizer_normalized = torch.optim.SGD(model_normalized.parameters(), lr=learning_rate)

trainer_normalized = Trainer(
    model=model_normalized,
    dataloader_train=dataloader_train_normalized,
    dataloader_test=dataloader_test_normalized,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer_normalized
)
train_loss_dict_normalized, test_loss_dict_normalized = trainer_normalized.train(num_epochs)


# We can now plot the training loss with our utility script

# Plot loss
utils.plot_loss(train_loss_dict, label="Train Loss")
utils.plot_loss(test_loss_dict, label="Test Loss")
# Task4 (a)
utils.plot_loss(train_loss_dict_normalized, label="Normalized Train Loss")
utils.plot_loss(test_loss_dict_normalized, label="Normalized Test Loss")

# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
plt.savefig("image_solutions/task_4c.png")

plt.show()

torch.save(model_normalized.state_dict(), "saved_model.torch")
final_loss_normalzied, final_acc_normalized = utils.compute_loss_and_accuracy(
    dataloader_test_normalized, model_normalized, loss_function)
avg_loss_normalized = sum(test_loss_dict_normalized.values()) / float(len(test_loss_dict_normalized))
print(f"Average test loss of original model: {avg_loss_normalized}.")
print(f"Final Test loss: {final_loss_normalzied}. Final Test accuracy: {final_acc_normalized}")


# Task4 (b)
weight = list(model_normalized.children())[1].weight.cpu().data
for num in range(10):
    num_weight = weight[num].reshape(28,28)
    plt.imshow(num_weight,cmap="gray")
    plt.show()
    


# Task4 (c) new learning rate:
torch.random.manual_seed(0)
np.random.seed(0)

learning_rate_4c = 1.0

image_transform_4c = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(0.5,0.5)
])
dataloader_train_4c, dataloader_test_4c = dataloaders.load_dataset(
    batch_size, image_transform_4c)
example_images_4c, _4c = next(iter(dataloader_train_4c))

model_4c = create_model()

# Define optimizer (Stochastic Gradient Descent)
optimizer_4c = torch.optim.SGD(model_4c.parameters(), lr=learning_rate_4c)

trainer_4c = Trainer(
    model=model_4c,
    dataloader_train=dataloader_test_4c,
    dataloader_test=dataloader_test_4c,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer_4c
)
train_loss_dict_4c, test_loss_dict_4c = trainer_4c.train(num_epochs)


# We can now plot the training loss with our utility script

# Plot loss
utils.plot_loss(train_loss_dict_4c, label="4c Train Loss")
utils.plot_loss(test_loss_dict_4c, label="4c Test Loss")

# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([1, 50])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
plt.savefig("image_solutions/task_4c.png")

plt.show()

torch.save(model_4c.state_dict(), "saved_model_4c.torch")
final_loss_4c, final_acc_4c = utils.compute_loss_and_accuracy(
    dataloader_test_4c, model_4c, loss_function)
avg_loss_4c = sum(test_loss_dict_4c.values()) / float(len(test_loss_dict_4c))
print(f"Average test loss of 4c: {avg_loss_4c}.")
print(f"Final Test loss: {final_loss_4c}. Final Test accuracy: {final_acc_4c}")


# Task 4 (d)
torch.random.manual_seed(0)
np.random.seed(0)

image_transform_4d = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
dataloader_train_4d, dataloader_test_4d = dataloaders.load_dataset(
    batch_size, image_transform_4d)
example_images_4d, _4d = next(iter(dataloader_train_4d))
print(f"The tensor containing the images has shape: {example_images_4d.shape} (batch size, number of color channels, height, width)",
      f"The maximum value in the image is {example_images_4d.max()}, minimum: {example_images_4d.min()}", sep="\n\t")

def create_model_4d():
    """
        Initializes the mode. Edit the code below if you would like to change the model.
    """
    model = nn.Sequential(
        
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        nn.ReLU(inplace=False),
        nn.Linear(28*28*1, 64),
        nn.Linear(64,10),
        
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model


model_4d = create_model_4d()


# Test if the model is able to do a single forward pass
example_images_4d = utils.to_cuda(example_images_4d)
output_4d = model_4d(example_images_4d)
print("Output shape:", output_4d.shape)
expected_shape_4d = (batch_size, 10)  # 10 since mnist has 10 different classes
assert output_4d.shape == expected_shape_4d,    f"Expected shape: {expected_shape_4d}, but got: {output_4d.shape}"

# Define optimizer (Stochastic Gradient Descent)
optimizer_4d = torch.optim.SGD(model_4d.parameters(),
                            lr=learning_rate)


trainer_4d = Trainer(
    model=model_4d,
    dataloader_train=dataloader_train_4d,
    dataloader_test=dataloader_test_4d,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer_4d
)
train_loss_dict_4d, test_loss_dict_4d = trainer_4d.train(num_epochs)

torch.save(model_4d.state_dict(), "saved_model.torch")
final_loss_4d, final_acc_4d = utils.compute_loss_and_accuracy(
    dataloader_test_4d, model_4d, loss_function)

# Plot loss
utils.plot_loss(train_loss_dict_normalized, label="4a Train Loss")
utils.plot_loss(test_loss_dict_normalized, label="4a Test Loss")
utils.plot_loss(train_loss_dict_4d, label="New Train Loss")
utils.plot_loss(test_loss_dict_4d, label="New Test Loss")

# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
plt.savefig("image_solutions/task_4d.png")

plt.show()

torch.save(model_4d.state_dict(), "saved_model_4d.torch")
final_loss_4d, final_acc_4d = utils.compute_loss_and_accuracy(
    dataloader_test_4d, model_4d, loss_function)
avg_loss_4d = sum(test_loss_dict_4d.values()) / float(len(test_loss_dict_4d))
print(f"Average test loss of original model: {avg_loss_4d}.")
print(f"Final Test loss: {final_loss_4d}. Final Test accuracy: {final_acc_4d}")
