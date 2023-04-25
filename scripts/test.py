import torch
import time

def test(model, test_loader, device, loss_fn, logger): 
    # test set accuracy
    start = time.time()

    with torch.no_grad():

        test_loss = 0
        test_accuracy = 0

        for test_X, test_y in test_loader:
            test_X, test_y = test_X.float().to(device), test_y.to(device)
            
            test_preds = model(test_X)
            iter_loss = loss_fn(test_preds, test_y)

            test_loss += iter_loss            
            iter_accuracy = ((test_preds.argmax(dim=1) == test_y).float().mean())
            test_accuracy += iter_accuracy

        test_accuracy = test_accuracy/len(test_loader)
        test_loss = test_loss / len(test_loader)

        logger.info("Test Loss: {:.4f}, Test accuracy: {:.4f}, Time taken: {}\n".format(test_loss, test_accuracy, time.time() - start))