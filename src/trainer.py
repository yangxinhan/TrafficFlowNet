import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
    def train(self, train_loader, val_loader):
        self.model.to(self.device)
        best_val_loss = float('inf')
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            train_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                images = batch['images'].to(self.device)
                # 只取第一個邊界框的類別作為圖片類別
                labels = torch.tensor([l[0] for l in batch['labels']]).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            val_loss = self.validate(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
        return val_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, f'model_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
