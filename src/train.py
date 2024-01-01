import os
import time
import datetime
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate
from dataset import BaseDataset
from config import NAMLConfig as config
from model.naml import NAML as Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """A lower value is better, e.g. loss."""
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def time_since(since):
    """Format elapsed time string."""
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    # Create saving directory for tensorboard
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    timestamp_for_path = timestamp.replace(":", "_")
    writer = SummaryWriter(
        log_dir=f"./runs/NAML/{timestamp_for_path}"
    )

    # Create saving directory for checkpoints
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load('./processed_data/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    model = Model(config, pretrained_word_embedding).to(device)

    dataset = BaseDataset('processed_data/train/transactions_parsed.csv',
                          'processed_data/articles_parsed.csv')

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=True))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config.learning_rate)

    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()

    checkpoint_dir = os.path.join('./checkpoint', 'NAML')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        try:
            early_stopping(checkpoint['early_stop_value'])
        except:
            pass
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()


    for i in tqdm(range(1, config.num_epochs * len(dataset) // config.batch_size + 1), desc="Training"):
        try:
            minibatch = next(dataloader)
        except StopIteration:
            exhaustion_count += 1
            tqdm.write(
                f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            )
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=True))
            minibatch = next(dataloader)

        step += 1
        candidates = minibatch["candidate_articles"]
        prev_purchased = minibatch["prev_purchased_parsed"]
        y_pred = model(candidates, prev_purchased)

        y_true = torch.zeros_like(y_pred).double().to(device)
        y_true[:, 0] = 1 # The first one is the positive sample
        
        loss = criterion(y_pred, y_true)
        loss_full.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), step)

        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )
            try:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                    }, f"./checkpoint/NAML/ckpt-{step}.pth")
            except OSError as error:
                print(f"OS error: {error}")

        if i % config.num_batches_validate == 0:
            model.eval()
            recall, val_apk12 = evaluate(model, './processed_data', 200000)
            
            model.train()
            writer.add_scalar('Validation/Recall', recall, step)
            writer.add_scalar('Validation/APK@12', val_apk12, step)
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, validation Recall: {recall:.4f}, validation APK@12: {val_apk12:.4f}"
            )

            early_stop, get_better = early_stopping(-recall)
            if early_stop:
                tqdm.write('Early stop.')
                break
            elif get_better:
                try:
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': step,
                            'early_stop_value': -recall
                        }, f"./checkpoint/NAML/ckpt-better-{step}.pth")
                except OSError as error:
                    print(f"OS error: {error}")


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training model NAML...')
    train()