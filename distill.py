import torch
import torchvision.datasets
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from models import TeacherModel, StudentModel
from utils import evaluate, load_data
from tqdm import tqdm
from loguru import logger

def distill():
    logger.add("logs/distilling.log", rotation="1 day")

    # seed everything
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_loader, test_loader = load_data()

    # 准备好预训练好的教师模型
    teacher_model = TeacherModel().eval().to(device)
    teacher_model.load_state_dict(torch.load('./checkpoints/teacher_model.pth'))
    logger.info('Teacher model loaded.')

    # 准备好新的学生模型
    model = StudentModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 蒸馏温度
    temperature = 7

    hard_loss = nn.CrossEntropyLoss()
    alpha = 0.3
    soft_loss = nn.KLDivLoss(reduction='batchmean')

    best_acc = 0
    for epoch in range(20):
        model.train()
        # 在训练集上训练权重
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training epoch {}'.format(epoch))):
            data, target = data.to(device), target.to(device)

            # 教师模型预测
            with torch.no_grad():
                teacher_preds = teacher_model(data)

            # 学生模型预测
            student_preds = model(data)

            # 计算hard loss
            student_loss = hard_loss(student_preds, target)
            # 计算soft loss
            distillation_loss = soft_loss(
                F.softmax(student_preds / temperature, dim=1),
                F.softmax(teacher_preds / temperature, dim=1)
            )

            # 计算总的loss
            loss = alpha * student_loss + (1 - alpha) * distillation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        accuracy = evaluate(test_loader, model, device)
        logger.info('Epoch: {} \tAccuracy: {:.4f}'.format(epoch, accuracy * 100))
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(teacher_model.state_dict(), './checkpoints/student_model.pth')
        logger.info('Best Accuracy: {:.4f}'.format(best_acc * 100))

if __name__ == '__main__':
    distill()


