import torch


def test(model, testloader, batch_size: int, classes: list):
    with torch.no_grad():
        n_correct, n_samples = 0, 0
        n_class_correct, n_class_samples = [i for i in range(10)], [
            i for i in range(10)
        ]

        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]

                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        accuracy = 100.0 * n_correct / n_samples
        print(f"\nAccuracy: {accuracy} %\n")

        for i in range(10):
            accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f"Accuracy of {classes[i]}: {accuracy} %")
