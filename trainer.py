import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from dataset import train_dataset , normalize
from model import Classifier

class multiclass_trainer():

    def __init__(self,args):

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.optim = args.optimizer
        self.normalize = args.normalize
        self.resume = args.resume
        self.start_epoch = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[!] torch version: {torch.__version__}")
        print(f"[!] computation device: {self.device}")

    def load_data(self):

        print("[!] Data Loading...")

        self.train_dataset = train_dataset(normalize() if self.normalize else None)
        
        # simple test
        # a , b = self.train_dataset[0]
        # print(type(a) , type(b))
        
        self.train_loader = DataLoader(dataset = self.train_dataset,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       num_workers = 1)
                                       
        print("[!] Data Loading Done.")


    def setup(self):

        print("[!] Setup...")

        # define our model, loss function, and optimizer

        self.Classifier = Classifier().to(self.device)


        
        if self.optim.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=self.lr , weight_decay = 1e-4)
        else:
            self.optimizer = torch.optim.SGD(self.Classifier.parameters(), lr=self.lr , weight_decay = 1e-4)

        self.criterion = nn.CrossEntropyLoss().to(self.device)


        # load checkpoint file to resume training
        if self.resume:
            print(f"[!] Resume training from the file : {self.resume}")
            checkpoint = torch.load(self.resume)
            self.Classifier.load_state_dict(checkpoint['model_state'])
            try:
                self.start_epoch = checkpoint['epoch']
            except:
                pass

        print("[!] Setup Done.")


    

    def train(self):

        print("[!] Model training...")

        n_total_steps = len(self.train_loader)
        n_total_samples = self.batch_size * n_total_steps

        for epoch in range(self.epochs):
            total_loss = 0
            running_loss = 0
            total_accuracy = 0
        
            for i , (inputs , targets) in enumerate(self.train_loader):

                # access data
                inputs = inputs.to(self.device)
                targets = targets.squeeze(1).to(self.device)

                # model feedforward
                outputs = self.Classifier(inputs)

                # compute loss
                loss = self.criterion(outputs , targets)

                # update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record loss and accuracy
                total_loss += loss.item() / n_total_steps
                running_loss += loss.item() / n_total_steps
                total_accuracy += (torch.argmax(outputs , dim = 1) == targets).sum().item() / n_total_samples

                

                if (i+1) % 100 == 0:
                    print(f"[!] Epoch : [{epoch}], step : [{i+1} / {n_total_steps}], Running Loss: {running_loss:.6f}")
                    running_loss = 0


            # per-epoch logging
            print("------------------------------------------")
            print(f"[!] Epoch : [{epoch+1}/{self.epochs}] , Loss: {total_loss:.6f}, Accuracy: {total_accuracy:.4f}\n")

        print("[!] Training Done.")

    
    def save(self):

        print("[!] Model saving...")

        checkpoint = {
                       "model_state": self.Classifier.state_dict(),
                     }

        torch.save(checkpoint , "checkpoint.pth")
    
        print("[!] Saving Done.")
