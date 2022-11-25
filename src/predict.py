import torch

from utils import device


def predict(model, dataloader):
    y_preds = []

    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            inputs = torch.squeeze(inputs, dim=1)
            y_test_pred = model(inputs)
            _, y_test_pred = torch.max(y_test_pred, 1)
            y_pred_tag = y_test_pred
            y_preds.append(y_pred_tag.cpu().numpy())

    y_preds = [x.squeeze().tolist() for x in y_preds]
    y_preds = [x for sublist in y_preds for x in sublist]

    return y_preds
