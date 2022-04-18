import torch

from torch import nn
import torch.optim as optim
from torchinfo import summary
from tqdm import tqdm
from models import TeacherModel, StudentModel
from utils import evaluate, load_data

def train():

    # seed everything
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用cuDNN加速卷积运算
    torch.backends.cudnn.benchmark = True

    # load data
    train_loader, test_loader = load_data()

    # build model
    model = StudentModel().to(device)
    summary(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0

    for epochs in range(20):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training epoch {}".format(epochs))):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        accuracy = evaluate(test_loader, model, device)
        print('Epoch: {} \tAccuracy: {:.4f}'.format(epochs, accuracy * 100))
        if accuracy > best_acc:
            best_acc = accuracy
            # torch.save(model.state_dict(), './checkpoints/model.pth')
        print('Best Accuracy: {:.4f}'.format(best_acc * 100))


if __name__ == '__main__':
    train()



