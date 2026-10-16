import torch

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": 100 * correct / total_examples,
    }