import torch
from torch import nn
from tqdm import tqdm
from Retrieval_Model.loss.Ranked_List import RankedLoss


def train_stage(model, train_loader, optimizer, scheduler, epochs=70):
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    rankloss = RankedLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        for data, label in tqdm(train_loader):
            model.train()
            data = data.cuda()
            label = label.cuda()

            fea, output = model(data)
            loss = criterion(output, label) + rankloss(fea, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")

    # save model
    torch.save(model.state_dict(), '../data/model/MixerMLP.pth')
