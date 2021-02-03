import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import torchvision.transforms as transforms

class lrcn(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):
        super(lrcn, self).__init__()
        self.conv = CNN_resnet(latent_dim)
        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        self.output_layer = nn.Linear(2 * hidden_size if bidirectional==True else hidden_size, n_class)

    def forward(self, x):
        x = x.transpose(1, 2)
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.reshape(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output = self.Lstm(lstm_input)
        output = torch.mean(self.output_layer(lstm_output), 1)

        return output

class CNN_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(CNN_resnet, self).__init__()
        self.conv = models.resnet152(pretrained=True)
        #self.conv = models.googlenet(pretrained=True)
        # ====== freezing all of the layers ======
        for param in self.conv.parameters():
            param.requires_grad = False
        # ====== changing the last FC layer to an output with the size we need. this layer is un freezed ======
        self.conv.fc = nn.Linear(self.conv.fc.in_features, latent_dim)

    def forward(self, x):
        return self.conv(x)

class Lstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        super(Lstm, self).__init__()
        self.Lstm = nn.LSTM(latent_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self,x):
        self.Lstm.flatten_parameters()
        output, self.hidden_state = self.Lstm(x, self.hidden_state)
        return output
    
def train_model(model, dataloader, device, optimizer, criterion):
    train_loss, train_acc = 0.0, 0.0
    model.train()
    with tqdm(total=len(dataloader)) as pbar:
        # with tqdm_notebook(total=len(dataloader)) as pbar:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            model.module.Lstm.reset_hidden_state()
            output = model(images)
            output = torch.log(torch.mean(output, dim=1))
            loss = criterion(output, labels)
            # Accuracy calculation
            predicted_labels = output.detach().argmax(dim=1)
            acc = (predicted_labels == labels).cpu().numpy().sum()
            train_loss += loss.item()
            train_acc += acc
            loss.backward()  # compute the gradients
            optimizer.step()  # update the parameters with the gradients
            pbar.update(1)
    train_acc = 100 * (train_acc / dataloader.dataset.__len__())
    train_loss = train_loss / len(dataloader)
    return train_loss, train_acc
              
def test_model(model, dataloader, device, criterion):
    val_loss, val_acc = 0.0, 0.0
    model.eval()
    with tqdm(total=len(dataloader)) as pbar:
        # with tqdm_notebook(total=len(dataloader)) as pbar:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            model.module.Lstm.reset_hidden_state()
            with torch.no_grad():
                output = model(images)
            #output = torch.log(output[:,-1,:])
            output = torch.log(torch.mean(output, dim=1))
            loss = criterion(output, labels)
            # Accuracy calculation
            predicted_labels = output.detach().argmax(dim=1)
            acc = (predicted_labels == labels).cpu().numpy().sum()
            val_loss += loss.item()
            val_acc += acc
            pbar.update(1)
    val_acc = 100 * (val_acc / dataloader.dataset.__len__())
    val_loss = val_loss / len(dataloader)
    return val_loss, val_acc, predicted_labels.cpu(), images.cpu()    

if __name__ == '__main__':
    model = lrcn(latent_dim=512, hidden_size=256, lstm_layers=1, bidirectional=True, n_class=101)
    input = torch.randn((2, 3, 10, 224, 224))
    output = model(input)
    print(output.shape)