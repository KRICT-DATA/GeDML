import random
import copy
import torch
from torch_geometric.data import Batch


def get_sorted_pairs(train_data):
    num_train_data = len(train_data)
    sorted_pairs = dict()

    sorted_train_data = sorted(copy.deepcopy(train_data), key=lambda x: x.y, reverse=True)
    for i in range(0, num_train_data):
        target = train_data[i].y

        sorted_pairs[i] = list()
        for j in range(0, num_train_data):
            if sorted_train_data[j].y > target:
                sorted_pairs[i].append(j)
            else:
                break

            print((i, j))

    return sorted_pairs


def get_pairs(data_list):
    num_data = len(data_list)
    pos_list = list()
    neg_list = list()

    for i in range(0, num_data):
        target = data_list[i].y
        idx = random.sample(range(0, num_data), 2)

        if abs(target - data_list[idx[0]].y) < abs(target - data_list[idx[1]].y):
            pos_list.append(data_list[idx[0]])
            neg_list.append(data_list[idx[1]])
        else:
            pos_list.append(data_list[idx[1]])
            neg_list.append(data_list[idx[0]])

    return Batch.from_data_list(pos_list), Batch.from_data_list(neg_list)


def get_pos_neg_pairs(data_list):
    num_data = len(data_list)
    pos_list = list()
    neg_list = list()
    labels = torch.tensor([x.y for x in data_list], dtype=torch.long)

    for i in range(0, num_data):
        same_labels = (labels == labels[i]).nonzero()
        pos_idx = same_labels[torch.randint(0, same_labels.shape[0], (1,)), 0]
        pos_list.append(data_list[pos_idx])

        diff_labels = (labels != labels[i]).nonzero()
        neg_idx = diff_labels[torch.randint(0, diff_labels.shape[0], (1,)), 0]
        neg_list.append(data_list[neg_idx])

    return Batch.from_data_list(pos_list), Batch.from_data_list(neg_list)


def train(model, optimizer, data_loader):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        batch_pos, batch_neg = get_pairs(batch.to_data_list())

        batch.batch = batch.batch.cuda()
        batch_pos.batch = batch_pos.batch.cuda()
        batch_neg.batch = batch_neg.batch.cuda()

        emb_anc = model(batch)
        emb_pos = model(batch_pos)
        emb_neg = model(batch_neg)

        dist_ratio_x = torch.norm(emb_anc - emb_pos, dim=1) / (torch.norm(emb_anc - emb_neg, dim=1) + 1e-5)
        dist_ratio_x = -(torch.exp(-dist_ratio_x + 1) - 1)
        dist_ratio_y = torch.norm(batch.y - batch_pos.y, dim=1) / (torch.norm(batch.y - batch_neg.y, dim=1) + 1e-5)
        dist_ratio_y = -(torch.exp(-dist_ratio_y + 1) - 1)

        loss = torch.mean((dist_ratio_x - dist_ratio_y)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

        if (i + 1) % 20 == 0:
            print('[' + str(i + 1) + '/' + str(len(data_loader)) + ']')

    return train_loss / len(data_loader)


def train_clf(model, optimizer, data_loader, alpha):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        batch_pos, batch_neg = get_pos_neg_pairs(batch.to_data_list())

        batch.batch = batch.batch.cuda()
        batch_pos.batch = batch_pos.batch.cuda()
        batch_neg.batch = batch_neg.batch.cuda()

        emb_anc = model(batch)
        emb_pos = model(batch_pos)
        emb_neg = model(batch_neg)

        dist_pos = torch.norm(emb_anc - emb_pos, dim=1)
        dist_neg = torch.norm(emb_anc - emb_neg, dim=1)
        loss = torch.mean(torch.clamp(alpha + dist_pos - dist_neg, min=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

        if (i + 1) % 20 == 0:
            print('[' + str(i + 1) + '/' + str(len(data_loader)) + ']')

    return train_loss / len(data_loader)


def test(model, data_loader):
    model.eval()
    emb_result = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            emb = model(batch)
            emb_result.append(emb)

    return torch.cat(emb_result, dim=0)
