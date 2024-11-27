# %%
import ssl

import lightning
import segmentation_models_pytorch as smp
import torch

import loss

from unet import Unet

# %%
# avoid error for downloading pre-trained model
ssl._create_default_https_context = ssl._create_unverified_context

# %%
ENCODER_NAME = "resnet50"


# %%
# pytorch model based on lighting
class ContrailModel(lightning.LightningModule):
    def __init__(self, arch, in_channels, out_classes, **kwargs):
        super().__init__()

        kwargs_ = kwargs.copy()
        if "loss" in kwargs_:
            kwargs_.pop("loss")

        if kwargs["pretrained"] is False:
            self.model = Unet(
                encoder_name="resnet50",
                in_channels=in_channels,
                classes=out_classes,
                encoder_weights=None,
            )
        else:
            if in_channels == 3:
                self.model = Unet(
                    encoder_name="resnet50",
                    in_channels=in_channels,
                    classes=out_classes,
                )
            else:
                self.model = Unet(
                    encoder_name=ENCODER_NAME,
                    in_channels=in_channels,
                    classes=out_classes,
                )

        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(ENCODER_NAME)
        self.register_buffer("std", torch.tensor(params["std"]).mean())
        self.register_buffer("mean", torch.tensor(params["mean"]).mean())

        loss_name = kwargs.get("loss", "dice").lower()

        if loss_name == "dice":
            self.loss_fn = loss.DiceLoss(log_loss=True)
        elif loss_name == "focal":
            self.loss_fn = loss.FocalLoss(normalized=False)
        elif loss_name == "sr":
            loss_base = kwargs.get("loss_base", "dice").lower()
            assert loss_base in ["dice", "focal"]
            self.loss_fn = loss.SRLoss(loss_base=loss_base)
        elif loss_name == 'recon':
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError

        self.pretrain = loss_name == 'recon'

        self.dice = loss.DiceLoss(log_loss=False)

        self.outputs = {"train": [], "valid": []}

    def forward(self, image):
        # normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # for grayscale images, expand channels dim to be [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32);
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        pred_mask = self.forward(image)

        if self.pretrain:
            loss = self.loss_fn(pred_mask, image)
            stats = {"loss": loss}
        else:
            loss = self.loss_fn(pred_mask, mask)
            dice = self.dice(pred_mask, mask)

            # Compute metrics for some threshold
            # first convert mask values to probabilities, then apply thresholding
            prob_mask = pred_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            # Compute true positive, false positive, false negative and true negative
            # of 'pixels' for each image and class. Aggregate them at epoch end.
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), mask.long(), mode="binary"
            )

            stats = {"loss": loss, "dice": dice, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

            self.outputs[stage].append(stats)

        return stats

    def shared_epoch_end(self, stage):
        if len(self.outputs[stage]) == 0:
            # happens when starting from checkpoint
            return

        # aggregate step metics
        tp = torch.cat([x["tp"] for x in self.outputs[stage]])
        fp = torch.cat([x["fp"] for x in self.outputs[stage]])
        fn = torch.cat([x["fn"] for x in self.outputs[stage]])
        tn = torch.cat([x["tn"] for x in self.outputs[stage]])

        # First calculate IoU score for each image, then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        per_image_dice = smp.metrics.f1_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # aggregate intersection and union over whole dataset and compute IoU score.
        # For dataset with "empty" images (images without target class), a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        dice_loss = torch.tensor([x["dice"] for x in self.outputs[stage]]).nanmean()
        model_loss = torch.tensor([x["loss"] for x in self.outputs[stage]]).nanmean()

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_per_image_dice": per_image_dice,
            f"{stage}_dataset_dice": dataset_dice,
            f"{stage}_dice_loss": dice_loss,
            f"{stage}_model_loss": model_loss,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        self.shared_epoch_end("train")
        self.outputs["train"].clear()

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        self.shared_epoch_end("valid")
        self.outputs["valid"].clear()

    def configure_optimizers(self):
        # try AdamW
        return torch.optim.Adam(self.parameters())
