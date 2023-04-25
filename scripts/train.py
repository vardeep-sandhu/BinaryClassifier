import torch
import time
from tqdm import tqdm 

def train(model, train_loader, device, val_loader, optimizer, loss_fn, args, logger):
    losses = []
    accuracies = []
    epoches = args.epochs
    start = time.time()
    for epoch in range(epoches):

    
        epoch_loss = 0
        epoch_accuracy = 0
        # progress_bar_train = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for X, y in train_loader:
            X, y = X.float().to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = ((preds.argmax(dim=1) == y).float().mean())
            epoch_accuracy += accuracy
            epoch_loss += loss
            logger.info("iter loss: {:.4f}, iter acc: {:.4f}".format(loss, accuracy))
            
        epoch_accuracy = epoch_accuracy/len(train_loader)
        accuracies.append(epoch_accuracy)
        epoch_loss = epoch_loss / len(train_loader)
        losses.append(epoch_loss)

        logger.info("\n --- Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(epoch + 1, epoch_loss, epoch_accuracy, time.time() - start))

        with torch.no_grad():

            test_epoch_loss = 0
            test_epoch_accuracy = 0
            # progress_bar_test = tqdm(enumerate(val_loader), total=len(val_loader))
        
            for test_X, test_y in val_loader:
                test_X, test_y = test_X.float().to(device), test_y.to(device)
                
                test_preds = model(test_X)
                test_loss = loss_fn(test_preds, test_y)

                test_epoch_loss += test_loss            
                test_accuracy = ((test_preds.argmax(dim=1) == test_y).float().mean())
                test_epoch_accuracy += test_accuracy

            test_epoch_accuracy = test_epoch_accuracy/len(val_loader)
            test_epoch_loss = test_epoch_loss / len(val_loader)

            logger.info("Epoch: {}, test loss: {:.4f}, test acc: {:.4f}, time: {}\n".format(epoch + 1, test_epoch_loss, test_epoch_accuracy, time.time() - start))
