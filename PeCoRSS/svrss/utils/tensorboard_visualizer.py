"""Class to create Tensorboard Visualization during training"""
from torch.utils.tensorboard import SummaryWriter


class TensorboardMultiLossVisualizer:
    """Class to generate Tensorboard visualisation

    PARAMETERS
    ----------
    writer: SummaryWriter from Tensorboard
    """

    def __init__(self, writer):
        self.writer = writer

    def update_train_loss(self, loss, losses, iteration):
        self.writer.add_scalar('train_losses/global', loss,
                               iteration)
        self.writer.add_scalar('train_losses/CE', losses[0],
                               iteration)
        self.writer.add_scalar('train_losses/Dice', losses[1],
                               iteration)

    def update_multi_train_loss(self, global_loss, rd_loss, rd_losses, iteration):
        self.writer.add_scalar('train_losses/global', global_loss,
                               iteration)
        self.writer.add_scalar('train_losses/range_doppler/global', rd_loss,
                               iteration)
        self.writer.add_scalar('train_losses/range_doppler/CE', rd_losses[0],
                               iteration)
        self.writer.add_scalar('train_losses/range_doppler/Dice', rd_losses[1],
                               iteration)

    def update_val_loss(self, loss, losses, iteration):
        self.writer.add_scalar('val_losses/global', loss,
                               iteration)
        self.writer.add_scalar('val_losses/CE', losses[0],
                               iteration)
        self.writer.add_scalar('val_losses/Dice', losses[1],
                               iteration)

    def update_multi_val_loss(self, global_loss, rd_loss, rd_losses, iteration):
        self.writer.add_scalar('validation_losses/global', global_loss,
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler/global', rd_loss,
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler/CE', rd_losses[0],
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler/Dice', rd_losses[1],
                               iteration)

    def update_learning_rate(self, lr, iteration):
        self.writer.add_scalar('parameters/learning_rate', lr, iteration)

    def update_val_metrics(self, metrics, iteration):
        self.writer.add_scalar('validation_losses/global', metrics['loss'],
                               iteration)
        self.writer.add_scalar('validation_losses/CE', metrics['loss_ce'],
                               iteration)
        self.writer.add_scalar('validation_losses/Dice', metrics['loss_dice'],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Mean', metrics['acc'],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Background',
                               metrics['acc_by_class'][0],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Pedestrian',
                               metrics['acc_by_class'][1],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Cyclist',
                               metrics['acc_by_class'][2],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Car',
                               metrics['acc_by_class'][3],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Mean', metrics['prec'],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Background',
                               metrics['prec_by_class'][0],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Pedestrian',
                               metrics['prec_by_class'][1],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Cyclist',
                               metrics['prec_by_class'][2],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Car',
                               metrics['prec_by_class'][3],
                               iteration)
        self.writer.add_scalar('PixelRecall/Mean', metrics['recall'],
                               iteration)
        self.writer.add_scalar('PixelRecall/Background',
                               metrics['recall_by_class'][0],
                               iteration)
        self.writer.add_scalar('PixelRecall/Pedestrian',
                               metrics['recall_by_class'][1],
                               iteration)
        self.writer.add_scalar('PixelRecall/Cyclist',
                               metrics['recall_by_class'][2],
                               iteration)
        self.writer.add_scalar('PixelRecall/Car',
                               metrics['recall_by_class'][3],
                               iteration)
        self.writer.add_scalar('MIoU/Mean', metrics['miou'],
                               iteration)
        self.writer.add_scalar('MIoU/Background',
                               metrics['miou_by_class'][0],
                               iteration)
        self.writer.add_scalar('MIoU/Pedestrian',
                               metrics['miou_by_class'][1],
                               iteration)
        self.writer.add_scalar('MIoU/Cyclist',
                               metrics['miou_by_class'][2],
                               iteration)
        self.writer.add_scalar('MIoU/Car',
                               metrics['miou_by_class'][3],
                               iteration)

    def update_multi_val_metrics(self, signal_type, metrics, iteration):
        self.writer.add_scalar('validation_losses/global',
                               metrics[signal_type]['loss'],
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler/global',
                               metrics[signal_type]['loss'], iteration)
        self.writer.add_scalar('validation_losses/range_doppler/CE',
                               metrics[signal_type]['loss_ce'], iteration)
        self.writer.add_scalar('validation_losses/range_doppler/Dice',
                               metrics[signal_type]['loss_dice'], iteration)

        self.writer.add_scalar('Range_Doppler_metrics/PixelAccuracy',
                               metrics[signal_type]['acc'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/PixelPrecision',
                               metrics[signal_type]['prec'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/PixelRecall',
                               metrics[signal_type]['recall'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/MIoU',
                               metrics[signal_type]['miou'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/Dice',
                               metrics[signal_type]['dice'],
                               iteration)

    def update_detection_val_metrics(self, metrics, iteration):
        self.writer.add_scalar('AveragePrecision/Mean', metrics['map'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Pedestrian',
                               metrics['map_by_class']['pedestrian'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Cyclist',
                               metrics['map_by_class']['cyclist'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Car',
                               metrics['map_by_class']['car'],
                               iteration)

    def update_multi_test_metrics(self, signal_type, metrics, iteration):
        self.writer.add_scalar('test_losses/global',
                               metrics[signal_type]['loss'],
                               iteration)
        self.writer.add_scalar('test_losses/range_doppler/global',
                               metrics[signal_type]['loss'], iteration)
        self.writer.add_scalar('test_losses/range_doppler/CE',
                               metrics[signal_type]['loss_ce'], iteration)
        self.writer.add_scalar('test_losses/range_doppler/Dice',
                               metrics[signal_type]['loss_dice'], iteration)

        self.writer.add_scalar('Test/Range_Doppler_metrics/PixelAccuracy',
                               metrics[signal_type]['acc'],
                               iteration)
        self.writer.add_scalar('Test/Range_Doppler_metrics/PixelPrecision',
                               metrics[signal_type]['prec'],
                               iteration)
        self.writer.add_scalar('Test/Range_Doppler_metrics/PixelRecall',
                               metrics[signal_type]['recall'],
                               iteration)
        self.writer.add_scalar('Test/Range_Doppler_metrics/MIoU',
                               metrics[signal_type]['miou'],
                               iteration)
        self.writer.add_scalar('Test/Range_Doppler_metrics/Dice',
                               metrics[signal_type]['dice'],
                               iteration)

    def update_img_masks(self, pred_grid, gt_grid, iteration):
        self.writer.add_image('Predicted_masks', pred_grid, iteration)
        self.writer.add_image('Ground_truth_masks', gt_grid, iteration)

    def update_multi_img_masks(self, rd_pred_grid, rd_gt_grid, iteration):
        self.writer.add_image('Range_Doppler/Predicted_masks', rd_pred_grid, iteration)
        self.writer.add_image('Range_Doppler/Ground_truth_masks', rd_gt_grid, iteration)
