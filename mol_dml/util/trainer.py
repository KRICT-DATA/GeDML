import torch
import numpy
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


def train(model, optimizer, data_loader, criterion, reg=True):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        batch.batch = batch.batch.cuda()

        pred, _ = model(batch)
        if reg:
            loss = criterion(batch.y, pred)
        else:
            loss = criterion(pred, batch.y.long().squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

        if (i + 1) % 20 == 0:
            print('[' + str(i + 1) + '/' + str(len(data_loader)) + ']')

    return train_loss / len(data_loader)


def test(model, data_loader, criterion):
    model.eval()
    test_loss = 0
    pred_result = list()
    emb_result = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            pred, emb = model(batch)
            test_loss += criterion(batch.y, pred).detach().item()
            pred_result.append(pred)
            emb_result.append(emb)

    return test_loss / (len(data_loader)), torch.cat(pred_result, dim=0), torch.cat(emb_result, dim=0)


def test_clf(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    pred_result = list()

    f1_pred = list()
    f1_target = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            pred = model(batch)
            _, predicted = torch.max(pred, 1)
            correct += (predicted.view(-1, 1) == batch.y.long()).sum().item()
            total += batch.y.shape[0]
            pred_result.append(pred)

            f1_pred.append(predicted.cpu().numpy())
            f1_target.append(batch.y.squeeze(1).cpu().numpy())

    accuracy = 100 * (correct / float(total))

    f1_pred = numpy.hstack(f1_pred)
    f1_target = numpy.hstack(f1_target)
    print(f1_score(f1_pred, f1_target, average='weighted'))
    print(matthews_corrcoef(f1_pred, f1_target))

    return accuracy, torch.cat(pred_result, dim=0)
