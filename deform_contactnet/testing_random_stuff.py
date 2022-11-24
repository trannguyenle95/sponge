import torch

from sklearn.metrics import f1_score   

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    
    epsilon = 1e-7
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)

    return f1,true_positives, false_positives, true_negatives, false_negatives
    
pred = [[0, 1],
        [1, 0],
        [0, 0]]
target = [[0, 0],
         [1, 0],
         [0, 0]]
pred = torch.tensor(pred)
target = torch.tensor(target)
# correct = (pred == target).float()
f1,tp, fp, tn, fn = confusion(pred, target)
print(f1, tp, fp, tn , fn)
# f1_score = f1_score(target, pred)

