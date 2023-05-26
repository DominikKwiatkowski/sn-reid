from __future__ import division, absolute_import
import torch
import torch.nn as nn

class CentroidTripletLoss(nn.Module):
    """
    Centroidtriplet loss with hard positive/negative mining.
    
    Reference:
        On the Unreasonable Effectiveness of Centroids in Image Retrieval. arXiv:2104.13643.
        
    Args:
        margin (float, optional): margin for centroidtriplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(CentroidTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        # Compute avaraage feature for each target class. Not all classes are present in a batch.
        # So, we need to find the classes that are present in a batch and compute the average feature for each of them.
        
        # Find the unique classes in a batch
        unique_classes = torch.unique(targets)
        # Create a tensor of zeros with shape (num_classes, feat_dim)
        average_features = torch.zeros((unique_classes.shape[0], inputs.shape[1])).to(inputs.device)
        # For each unique class, find the average feature
        for i, unique_class in enumerate(unique_classes):
            # Find the indexes of the unique class in the batch
            indexes = torch.where(targets == unique_class)
            # Find the average feature for the unique class
            average_feature = torch.mean(inputs[indexes], dim=0)
            # Store the average feature for the unique class
            average_features[i] = average_feature
        
        # Now, compute avarage of all classes "rest". "rest" is the average of all classes except the current class.
        average_rest_features = torch.zeros((unique_classes.shape[0], inputs.shape[1])).to(inputs.device)
        for i, unique_class in enumerate(unique_classes):
            average_rest_features[i] = torch.mean(average_features[unique_classes != i], dim=0)
        
        # for each anchor, calculate the distance to the average feature of its class and the average feature of all other classes
        dist_ap, dist_an = [], []
        for i in range(inputs.shape[0]):
            index = torch.where(unique_classes == targets[i])[0]
            dist_ap.append(torch.dist(inputs[i], average_features[index]).unsqueeze(0))
            dist_an.append(torch.dist(inputs[i], average_rest_features[index]).unsqueeze(0))
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


        

        