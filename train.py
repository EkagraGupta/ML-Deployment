import torch
from hyperparameters import PATH


def train(model, criterion, optimizer, trainloader, n_epochs, path: str = PATH):
    print("Started Training...")
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(trainloader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print(
                    f"Epoch[{epoch + 1}/{n_epochs}], Step[{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}"
                )
    print("Finished Training")

    torch.save(model.state_dict(), path)
    print(f'Model saved under "{path}"')
