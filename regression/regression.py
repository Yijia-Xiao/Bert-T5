import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
import argparse
import scipy.stats
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--model-type', type=str)
parser.add_argument('--mean', action='store_true')

# parser.add_argument('--split', type=str)
args = parser.parse_args()

model_type = args.model_type
# writer = SummaryWriter(log_dir=f'./logs/{model_type}')
# writer = SummaryWriter(log_dir=f'./logs/{model_type}-bs32-1e-4')

if model_type == 'esm1b':
    embed_size = 1280
    train_path = f"/dataset/ee84df8b/yijia/esm/esm1b_train.pt"
    valid_path = f"/dataset/ee84df8b/yijia/esm/esm1b_valid.pt"
    test_path = f"/dataset/ee84df8b/yijia/esm/esm1b_test.pt"
    use_mean = args.mean
elif model_type == 't5-xl' or model_type == 't5-xxl':
    embed_size = 1024
    train_path = f"/dataset/ee84df8b/ProtTrans/data/prot_{model_type.replace('-', '_')}_uniref50_train.pt"
    valid_path = f"/dataset/ee84df8b/ProtTrans/data/prot_{model_type.replace('-', '_')}_uniref50_valid.pt"
    test_path = f"/dataset/ee84df8b/ProtTrans/data/prot_{model_type.replace('-', '_')}_uniref50_test.pt"

writer = SummaryWriter(
    log_dir=f'./logs/{model_type}-mean-{use_mean}-bs32-1e-4')


def data_prodiver(model_type: str):
    assert model_type in ['esm1b', 't5-xl', 't5-xxl']
    raw_train = torch.load(train_path)
    raw_valid = torch.load(valid_path)
    raw_test = torch.load(test_path)

    if model_type == 'esm1b':
        # esm[0]['embed']['mean_representations'][33]
        # use_cls = False
        if use_mean:
            print('use mean of BERT\'s token representations')
            train_data = [(i['embed']['mean_representations'][33],
                           i['log_fluorescence']) for i in raw_train]
            valid_data = [(i['embed']['mean_representations'][33],
                           i['log_fluorescence']) for i in raw_valid]
            test_data = [(i['embed']['mean_representations'][33],
                          i['log_fluorescence']) for i in raw_test]
        else:
            train_data = [(i['embed']['bos_representations'][33],
                           i['log_fluorescence']) for i in raw_train]
            valid_data = [(i['embed']['bos_representations'][33],
                           i['log_fluorescence']) for i in raw_valid]
            test_data = [(i['embed']['bos_representations'][33],
                          i['log_fluorescence']) for i in raw_test]

    elif model_type == 't5-xl' or model_type == 't5-xxl':
        train_data = [(i['embed'].mean(axis=0), i['log_fluorescence'])
                      for i in raw_train]
        valid_data = [(i['embed'].mean(axis=0), i['log_fluorescence'])
                      for i in raw_valid]
        test_data = [(i['embed'].mean(axis=0), i['log_fluorescence'])
                     for i in raw_test]
    return train_data, valid_data, test_data


class FluorescenceData(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        # self.feat = [i[0].clone().detach() for i in self.data]
        self.feat = [i[0] for i in self.data]
        self.labl = [torch.tensor(i[1]) for i in self.data]
        # self.feat = [i[0] for i in self.data]
        # self.labl = [torch.tensor(i[1]) for i in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.feat[index], self.labl[index]


class RegressionHead(nn.Module):
    def __init__(self, model_type, hidden_size=[512, 128]) -> None:
        super().__init__()
        self.net = nn.Sequential()
        self.hidden = hidden_size
        self.hidden.insert(0, embed_size)
        self.hidden.append(1)
        for idx in range(len(self.hidden) - 1):
            self.net.add_module(f'fc_{idx}', nn.Linear(
                self.hidden[idx], self.hidden[idx + 1]))
            self.net.add_module(f'act_{idx}', nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


def spearmanr_eval(pred, fact):
    pred_all = [i.item() for batch in pred for i in batch]
    fact_all = [i.item() for batch in fact for i in batch]
    spearmanr_score = scipy.stats.spearmanr(pred_all, fact_all).correlation
    return spearmanr_score


def main():
    train_data, valid_data, test_data = data_prodiver(model_type)
    for d in [train_data, valid_data, test_data]:
        print(len(d))

    train_set = FluorescenceData(train_data)
    valid_set = FluorescenceData(valid_data)
    test_set = FluorescenceData(test_data)
    num_works = 32
    train_loader = DataLoader(
        dataset=train_set, batch_size=num_works, shuffle=True, num_workers=num_works)
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=num_works, shuffle=False, num_workers=num_works)
    test_loader = DataLoader(
        dataset=test_set, batch_size=num_works, shuffle=False, num_workers=num_works)

    model = RegressionHead(model_type).to('cuda')
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    NUM_EPOCH = 500
    tot_steps = 0
    for ep in range(NUM_EPOCH):
        running_loss = 0
        train_steps = 0
        # train
        model.train()
        train_pred = []
        train_fact = []
        for data, label in train_loader:
            optimizer.zero_grad()
            pred = model(data.to('cuda'))
            loss = loss_fn(pred, label.to('cuda'))
            # pred = model(data)
            # loss = loss_fn(pred, label)
            # print(loss)
            running_loss += loss.item()
            train_steps += 1
            tot_steps += 1
            writer.add_scalar('train-loss-iter', loss.item(), tot_steps)

            loss.backward()
            optimizer.step()
            # print(pred, label)
            train_pred.append(pred.clone().detach())
            train_fact.append(label.clone().detach())
        writer.add_scalar('train-loss-epoch',
                          running_loss / train_steps, ep + 1)

        train_spear = spearmanr_eval(train_pred, train_fact)
        print('train:', train_spear)
        writer.add_scalar('train-spearmanr', train_spear, ep + 1)

        model.eval()
        # validation
        valid_loss = 0
        valid_steps = 0
        valid_pred = []
        valid_fact = []
        for data, label in valid_loader:
            pred = model(data.to('cuda'))
            loss = loss_fn(pred, label.to('cuda'))
            valid_loss += loss.item()
            valid_steps += 1
            valid_pred.append(pred.clone().detach())
            valid_fact.append(label.clone().detach())
        writer.add_scalar('valid-loss-epoch',
                          running_loss / valid_steps, ep + 1)
        valid_spear = spearmanr_eval(valid_pred, valid_fact)
        print('valid:', valid_spear)
        writer.add_scalar('valid-spearmanr', valid_spear, ep + 1)

        # test
        test_pred = []
        test_fact = []
        for data, label in test_loader:
            pred = model(data.to('cuda'))
            loss = loss_fn(pred, label.to('cuda'))
            test_pred.append(pred.clone().detach())
            test_fact.append(label.clone().detach())
        test_spear = spearmanr_eval(test_pred, test_fact)
        print('test:', test_spear)
        writer.add_scalar('test-spearmanr', test_spear, ep + 1)

    writer.close()


main()
