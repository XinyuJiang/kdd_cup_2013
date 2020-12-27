import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.datasets import load_svmlight_file
import config, os
import model_trainer.mod_cnn as mod_cnn
import model_trainer.evalution as evalution
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm


def prepare_data():
    sample_count = config.sample_count
    feature_count = config.feature_count
    batch_size = config.batch_size
    test_sample_count = config.test_sample_count

    #import train data
    # X = np.zeros((sample_count, feature_count, 1))
    X = np.zeros((sample_count, 1, feature_count))
    test_X, test_y = load_svmlight_file(os.path.join(config.CWD, "feature", "train.feature"))
    Y = test_y
    count = 0
    for i in test_X:
        for j in range(feature_count):
            # X[count, j, 0] = i[0, j]
            X[count, 0, j] = i[0, j]
        count += 1
    # print(X, Y)
    train_dataset = Data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    print(torch.tensor(X).shape)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    # print (train_dataset,train_loader)

    # import test data
    X = np.zeros((test_sample_count,  1, feature_count,))
    test_X, test_y = load_svmlight_file(os.path.join(config.CWD, "feature", "test.feature"))
    Y = test_y
    count = 0
    for i in test_X:
        for j in range(feature_count):
            X[count, 0, j] = i[0, j]
        count += 1
    test_dataset = Data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, len(train_dataset), test_loader, len(test_dataset)

if __name__ == "__main__":
    epoch = config.epoch
    batch_size = config.batch_size
    learn_rate = config.learn_rate
    weight_decay = config.weight_decay
    mode_save_path = config.mode_save_path

    # prepare data
    train_loader, train_count, test_loader, test_count = prepare_data()

    #train process
    #model get
    model = mod_cnn.CNN()
    model = model.cuda()

    #loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    #clean cuda
    torch.cuda.empty_cache()
    print('Start Train Process\n')
    for epoch in range(epoch):
        #init
        print('Epoch:', epoch+1)
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for i, data in tqdm(enumerate(train_loader, 1)):
            #use GPU
            item, label = data
            item = item.type(torch.FloatTensor)
            label = label.type(torch.LongTensor)
            item = item.cuda()
            label = label.cuda()

            #forward
            result = model(item)
            loss = criterion(result, label)
            epoch_loss += loss.item() * label.size(0)

            #calculate
            _, pred = torch.max(result, 1)
            print(pred)
            print(label)
            correct = (pred == label).sum()
            epoch_accuracy += correct.item()

            #back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print
            print('{} epoch, {} batch, Loss: {:.6f}, Accuracy: {:.6f}'.format(
                epoch + 1, i, loss.item() * label.size(0) / (len(label)), correct.item() / (len(label))))

        #train result
        print('Finish {} epoch, Loss: {:.6f}, Accuracy: {:.6f}'.format(
            epoch + 1, epoch_loss / train_count, epoch_accuracy / train_count))
    #save model
    torch.save(model.state_dict(), mode_save_path)
    print('Train Process End\n')


    # Test Validate
    print('Start Test Validate Process\n')
    pre_y = np.array([1])
    # init
    # use BN Layer, need .eval()
    model = mod_cnn.CNN()
    model.load_state_dict(torch.load(mode_save_path))
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    # clean cuda
    torch.cuda.empty_cache()
    print('Start  Validate Process\n')
    for i, data in tqdm(enumerate(test_loader, 1)):
        # use GPU
        item, label = data
        with torch.no_grad():
            item = item.type(torch.FloatTensor)
        with torch.no_grad():
            label = label.type(torch.LongTensor)

        # forward
        result = model(item)
        # loss = criterion(result, label)
        # val_loss += loss.item() * label.size(0)

        # calculate
        _, pred = torch.max(result, 1)
        pre_y = np.append(pre_y, pred)
        print(pred)

    predict_re = pre_y[1:]
    # print(predict_re)
    # print(len(predict_re))
    # write prediction to file
    result_file_path = os.path.join(config.CWD, "predict", "cnntest.result")
    with open(result_file_path, 'w') as fout:
        fout.write("\n".join(map(str, map(int, predict_re))))
    #change predict format
    test_feature_path = os.path.join(config.CWD, "feature", "test.feature")
    test_predict_path = os.path.join(config.CWD, "predict", "cnntest.predict")
    evalution.get_prediction(test_feature_path, result_file_path, test_predict_path)
    #evalute
    cmd = "python evalution.py %s %s" % (config.GOLD_FILE, test_predict_path)
    os.system(cmd)
    print('Validate Test Validate End\n')