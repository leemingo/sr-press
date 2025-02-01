"""Implements the SoccerMap architecture."""

from typing import Any, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class _FeatureExtractionLayer(nn.Module):
    """The 2D-convolutional feature extraction layer of the SoccerMap architecture.

    The probability at a single location is influenced by the information we
    have of nearby pixels. Therefore, convolutional filters are used for
    spatial feature extraction.

    Two layers of 2D convolutional filters with a 5 × 5 receptive field and
    stride of 1 are applied, each one followed by a ReLu activation function.
    To keep the same dimensions after the convolutional filters, symmetric
    padding is applied. It fills the padding cells with values that are
    similar to those around it, thus avoiding border-image artifacts that can
    hinder the model’s predicting ability and visual representation.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding="valid")
        # (left, right, top, bottom)
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.symmetric_padding(x)
        x = F.relu(self.conv_2(x))
        x = self.symmetric_padding(x)
        return x


class _PredictionLayer(nn.Module):
    """The prediction layer of the SoccerMap architecture.

    The prediction layer consists of a stack of two convolutional layers, the
    first with 32 1x1 convolutional filters followed by an ReLu activation
    layer, and the second consists of one 1x1 convolutional filter followed by
    a linear activation layer. The spatial dimensions are kept at each step
    and 1x1 convolutions are used to produce predictions at each location.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)  # linear activation
        return x


class _UpSamplingLayer(nn.Module):
    """The upsampling layer of the SoccerMap architecture.

    The upsampling layer provides non-linear upsampling by first applying a 2x
    nearest neighbor upsampling and then two layers of convolutional filters.
    The first convolutional layer consists of 32 filters with a 3x3 activation
    field and stride 1, followed by a ReLu activation layer. The second layer
    consists of 1 layer with a 3x3 activation field and stride 1, followed by
    a linear activation layer. This upsampling strategy has been shown to
    provide smoother outputs.
    """

    def __init__(self):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=1, padding="valid")
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = self.symmetric_padding(x)
        x = self.conv2(x)  # linear activation
        x = self.symmetric_padding(x)
        return x


class _FusionLayer(nn.Module):
    """The fusion layer of the SoccerMap architecture.

    The fusion layer merges the final prediction surfaces at different scales
    to produce a final prediction. It concatenates the pair of matrices and
    passes them through a convolutional layer of one 1x1 filter.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1)

    def forward(self, x: List[torch.Tensor]):
        out = self.conv(torch.cat(x, dim=1))  # linear activation
        return out


class SoccerMap(nn.Module):
    """SoccerMap architecture.

    SoccerMap is a deep learning architecture that is capable of estimating
    full probability surfaces for pass probability, pass slection likelihood
    and pass expected values from spatiotemporal data.

    The input consists of a stack of c matrices of size lxh, each representing a
    subset of the available spatiotemporal information in the current
    gamestate. The specific choice of information for each of these c slices
    might vary depending on the problem being solved

    Parameters
    ----------
    in_channels : int, default: 13
        The number of spatiotemporal input channels.

    References
    ----------
    .. [1] Fernández, Javier, and Luke Bornn. "Soccermap: A deep learning
       architecture for visually-interpretable analysis in soccer." Joint
       European Conference on Machine Learning and Knowledge Discovery in
       Databases. Springer, Cham, 2020.
    """

    def __init__(self, model_config):
        super().__init__()

        self.in_channels = model_config["in_channels"] * 3

        # Convolutions for feature extraction at 1x, 1/2x and 1/4x scale
        self.features_x1 = _FeatureExtractionLayer(self.in_channels)
        self.features_x2 = _FeatureExtractionLayer(64)
        self.features_x4 = _FeatureExtractionLayer(64)
        self.features_x8 = _FeatureExtractionLayer(64)
        self.features_x16 = _FeatureExtractionLayer(64)

        # Layers for down and upscaling and merging scales
        self.up_x2 = _UpSamplingLayer()
        self.up_x4 = _UpSamplingLayer()
        self.down_x2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x8 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x16 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fusion_x2_x4 = _FusionLayer()
        self.fusion_x1_x2 = _FusionLayer()

        # Prediction layers at each scale
        self.prediction_x1 = _PredictionLayer()
        self.prediction_x2 = _PredictionLayer()
        self.prediction_x4 = _PredictionLayer()
        self.prediction_x8 = _PredictionLayer()
        self.prediction_x16 = _PredictionLayer()

        # output layer: binary classification
        self.output_layer = nn.Sequential(
            nn.Flatten(),  # Flatten to (batch_size, num_features)
            nn.Linear((68 // 16) * (104 // 16), 1),  # Linear layer to output a single value
        )

    def forward(self, x):
        # Feature extraction
        f_x1 = self.features_x1(x)
        f_x2 = self.features_x2(self.down_x2(f_x1))
        f_x4 = self.features_x4(self.down_x4(f_x2))
        f_x8 = self.features_x8(self.down_x8(f_x4))
        f_x16 = self.features_x16(self.down_x16(f_x8))

        pred_x16 = self.prediction_x16(f_x16)

        # Prediction
        # pred_x1 = self.prediction_x1(f_x1)
        # pred_x2 = self.prediction_x2(f_x2)
        # pred_x4 = self.prediction_x4(f_x4)

        # Fusion
        # x4x2combined = self.fusion_x2_x4([self.up_x4(pred_x4), pred_x2])
        # combined = self.fusion_x1_x2([self.up_x2(x4x2combined), pred_x1]) # (bs, 1, 68, 104)

        # The activation function depends on the problem
        return self.output_layer(pred_x16)  # Output shape: (bs, 1)
        # return combined


def pixel(surface, mask):
    """Return the prediction at a single pixel.

    This custom layer is used to evaluate the loss at the pass destination.

    Parameters
    ----------
    surface : torch.Tensor
        The final prediction surface.
    mask : torch.Tensor
        A sparse spatial representation of the final pass destination.

    Returns
    -------
    torch.Tensor
        The prediction at the cell on the surface that matches the actual
        pass destination.
    """
    masked = surface * mask
    value = torch.sum(masked, dim=(3, 2))
    return value


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean"):
        """
        Focal Loss implementation.

        Args:
            alpha (float): Weighting factor for the positive class.
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predicted probabilities (output of sigmoid).
            targets (Tensor): Ground truth labels (binary, 0 or 1).

        Returns:
            Tensor: Computed focal loss.
        """
        # Clip inputs to avoid log(0)
        inputs = torch.clamp(inputs, 1e-8, 1 - 1e-8)

        # Compute the binary cross entropy loss
        bce_loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))

        # Compute the modulating factor
        pt = torch.where(targets == 1, inputs, 1 - inputs)  # p_t
        modulating_factor = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Compute the focal loss
        focal_loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class PytorchSoccerMapModel(pl.LightningModule):
    """A pass success probability model based on the SoccerMap architecture."""

    def __init__(self, model_config: dict = None, optimizer_params: dict = None):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SoccerMap(model_config=model_config)
        self.sigmoid = nn.Sigmoid()

        # loss function
        self.criterion = torch.nn.BCELoss()
        # self.criterion = FocalLoss()

        self.optimizer_params = optimizer_params

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.sigmoid(x)

        return x

    def step(self, batch: Any):
        x, y = batch
        y_hat = self.forward(x)

        # x, mask, y = batch
        # surface = self.forward(x)
        # y_hat = pixel(surface, mask)

        loss = self.criterion(y_hat, y)

        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch
        # x, mask, y = batch

        preds = self(x)  # self.forward(x)

        return preds, y

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return torch.optim.Adam(self.parameters(), **self.optimizer_params["optimizer_params"])


class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104), label=["concede_shots"]):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim
        self.label = label[0]

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):

        nb_prev_actions = 3

        num_features = 7
        # Output
        matrix = np.zeros((num_features * nb_prev_actions, self.y_bins, self.x_bins))

        for i in range(nb_prev_actions):
            start_x, start_y = sample[f"start_x_a{i}"], sample[f"start_y_a{i}"]

            if sample[f"freeze_frame_360_a{i}"] is None:
                continue

            frame = pd.DataFrame.from_records(sample[f"freeze_frame_360_a{i}"])
            target = int(sample[self.label]) if self.label in sample else None

            # Location of the player that passes the ball
            presser_coo = frame.loc[frame.actor, ["x", "y"]].fillna(1e-10).values.reshape(-1, 2)
            # Location of the ball
            ball_coo = np.array([[start_x, start_y]])
            # Location of the goal
            goal_coo = np.array([[105, 34]])
            # Locations of the passing player's teammates
            players_att_coo = frame.loc[frame.teammate, ["x", "y"]].values.reshape(-1, 2)

            # Locations and speed vector of the defending players
            players_def_coo = frame.loc[~frame.teammate, ["x", "y"]].values.reshape(-1, 2)

            x_bin_press, y_bin_press = self._get_cell_indexes(
                presser_coo[:, 0],
                presser_coo[:, 1],
            )
            matrix[0 + i * num_features, y_bin_press, x_bin_press] = 1

            # CH 1: Locations of attacking team
            x_bin_att, y_bin_att = self._get_cell_indexes(
                players_att_coo[:, 0],
                players_att_coo[:, 1],
            )
            matrix[1 + i * num_features, y_bin_att, x_bin_att] = 1

            # CH 2: Locations of defending team
            x_bin_def, y_bin_def = self._get_cell_indexes(
                players_def_coo[:, 0],
                players_def_coo[:, 1],
            )
            matrix[2 + i * num_features, y_bin_def, x_bin_def] = 1

            # CH 3: Distance to ball
            yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

            x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
            # matrix[3 + i * num_features, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

            # CH 4: Distance to goal
            x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
            # matrix[4 + i * num_features, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

            # CH 5: Distance to pressor
            # matrix[5 + i * num_features, :, :] = np.sqrt((xx - x_bin_press) ** 2 + (yy - y_bin_press) ** 2)
            matrix[3 + i * num_features, y0_ball, x0_ball] = 1

            # CH 6: Cosine of the angle between the ball and goal
            coords = np.dstack(np.meshgrid(xx, yy))
            goal_coo_bin = np.concatenate((x0_goal, y0_goal))
            ball_coo_bin = np.concatenate((x0_ball, y0_ball))
            a = goal_coo_bin - coords
            b = ball_coo_bin - coords
            # matrix[4 + i * num_features, :, :] = np.clip(
            #    np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
            # )

            # CH 7: Sine of the angle between the ball and goal
            sin = np.cross(a, b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
            # matrix[5 + i * num_features, :, :] = np.sqrt(1 - matrix[4 + i * num_features, :, :] ** 2)  # This is much faster

            # CH 8: Angle (in radians) to the goal location
            # matrix[6 + i * num_features, :, :] = np.abs(
            #    np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
            # )

        if target is not None:
            return (
                torch.from_numpy(matrix).float(),
                torch.tensor([target]).float(),
            )

        # simulated features: not exist labels
        return (
            torch.from_numpy(matrix).float(),
            None,
        )

        # Mask
        # mask = np.zeros((1, self.y_bins, self.x_bins))
        # start_x_idx, start_y_idx = self._get_cell_indexes(start_x, start_y)
        # mask[0, start_y_idx, start_x_idx] = 1

        # if target is not None:
        #     return (
        #         torch.from_numpy(matrix).float(),
        #         torch.from_numpy(mask).float(),
        #         torch.tensor([target]).float(),
        #     )

        # # simulated features: not exist labels
        # return (
        #     torch.from_numpy(matrix).float(),
        #     torch.from_numpy(mask).float(),
        #     None
        # )
