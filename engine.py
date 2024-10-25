"""
Train and eval functions used in main.py
"""
from typing import Iterable

import torch
import util.misc as util
import cv2
from datasets.data_prefetcher import data_prefetcher
import utils
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image


def calculate_metrics(y_true, y_pred):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return tn, fp, fn, tp

def calculate_final_metrics(tn, fp, fn, tp):
    # 计算各度量
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = recall = tp / (tp + fn)  # Recall 和 Sensitivity 是一样的
    specificity = tn / (tn + fp)
    ppv = precision = tp / (tp + fp)  # Precision 和 PPV 是一样的
    npv = tn / (tn + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1_score": f1_score
    }

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None, global_steps=0):
    model.train()
    criterion.train()
    metric_logger = util.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    tns = 0
    fps = 0
    fns = 0
    tps = 0
    accs_ceus_avg = []
    miou_ceus_avg = []
    mdice_ceus_avg = []
    accs_us_avg = []
    miou_us_avg = []
    mdice_us_avg = []
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for i in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples.tensors)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if writer is not None:
            writer.add_scalar('loss_sum_Current_epoch', losses, global_step=global_steps)
            writer.add_scalar('loss_mask_Current_epoch', loss_dict['loss_mask'], global_step=global_steps)
            writer.add_scalar('loss_ce_Current_epoch', loss_dict['loss_ce'], global_step=global_steps)
        global_steps += 1

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        with torch.no_grad():
            src_logits = outputs['label'].sigmoid().cpu()
            labels = []
            for t in targets:
                labels.append(t['labels'])
            target_classes_onehot = torch.stack(labels).cpu()

            zeros = torch.zeros_like(src_logits)
            ones = torch.ones_like(src_logits)
            src_logits = torch.where(src_logits >= 0.5, ones, zeros)

            tn, fp, fn, tp = calculate_metrics(target_classes_onehot.squeeze(1).numpy().tolist(), src_logits.squeeze(1).numpy().tolist())
            tns += tn
            fps += fp
            fns += fn
            tps += tp

            src_masks_ceus = outputs["seg_ceus"]
            target_masks_ceus = []
            for t in targets:
                target_masks_ceus.append(t['masks_ceus'])
            target_masks_ceus = torch.stack(target_masks_ceus).squeeze(1)

            accs_ceus, miou_ceus, mdice_ceus, kappas_ceus = utils.metrics.calculate_metrics(src_masks_ceus.cpu(), target_masks_ceus.cpu())
            accs_ceus_avg.append(accs_ceus)
            miou_ceus_avg.append(miou_ceus)
            mdice_ceus_avg.append(mdice_ceus)


            src_masks_us = outputs["seg_us"]
            target_masks_us = []
            for t in targets:
                target_masks_us.append(t['masks_us'])
            target_masks_us = torch.stack(target_masks_us).squeeze(1)

            accs_us, miou_us, mdice_us, kappas_us = utils.metrics.calculate_metrics(src_masks_us.cpu(), target_masks_us.cpu())
            accs_us_avg.append(accs_us)
            miou_us_avg.append(miou_us)
            mdice_us_avg.append(mdice_us)

        samples, targets = prefetcher.next()

    labels_metrics = calculate_final_metrics(tns, fps, fns, tps)
    traininfo1 = "Training: acc: %.4f | sens: %.4f | spec: %.4f | ppv: %.4f | npv: %.4f | f1: %.4f" \
                % (labels_metrics["accuracy"],
                   labels_metrics["sensitivity"],
                   labels_metrics["specificity"],
                   labels_metrics["ppv"],
                   labels_metrics["npv"],
                   labels_metrics["f1_score"])
    print(traininfo1)
    traininfo2 = "Training: acc_mask_ceus: %.4f | miou_ceus: %.4f | mdice_ceus: %.4f" \
                % (sum(accs_ceus_avg) / len(accs_ceus_avg), sum(miou_ceus_avg) / len(miou_ceus_avg), sum(mdice_ceus_avg) / len(mdice_ceus_avg))
    print(traininfo2)
    traininfo3 = "Training: acc_mask_us: %.4f | miou_us: %.4f | mdicus: %.4f" \
                % (sum(accs_us_avg) / len(accs_us_avg), sum(miou_us_avg) / len(miou_us_avg), sum(mdice_us_avg) / len(mdice_us_avg))
    print(traininfo3)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_steps


@torch.no_grad()
def evaluate(model, data_loader, device, val_steps, output_dir):
    model.eval()

    metric_logger = util.MetricLogger(delimiter="  ")
    header = 'Test:'

    accs_ceus_avg = []
    miou_ceus_avg = []
    mdice_ceus_avg = []
    accs_us_avg = []
    miou_us_avg = []
    mdice_us_avg = []
    tns = 0
    fps = 0
    fns = 0
    tps = 0
    for samples, targets in metric_logger.log_every(data_loader, 20, header):
        with torch.no_grad():
            samples = samples.to(device)
            outputs = model(samples.tensors)

            src_logits = outputs['label'].sigmoid().cpu()
            labels = []
            for t in targets:
                labels.append(t['labels'])
            target_classes_onehot = torch.stack(labels).cpu()

            zeros = torch.zeros_like(src_logits)
            ones = torch.ones_like(src_logits)
            src_logits = torch.where(src_logits >= 0.5, ones, zeros)


            tn, fp, fn, tp = calculate_metrics(target_classes_onehot.squeeze(1).numpy().tolist(), src_logits.squeeze(1).numpy().tolist())
            tns += tn
            fps += fp
            fns += fn
            tps += tp

            src_masks_ceus = outputs["seg_ceus"]
            target_masks_ceus = []
            for t in targets:
                target_masks_ceus.append(t['masks_ceus'])
            target_masks_ceus = torch.stack(target_masks_ceus).squeeze(1)

            accs_ceus, miou_ceus, mdice_ceus, kappas_ceus = utils.metrics.calculate_metrics(src_masks_ceus.cpu(),
                                                                                            target_masks_ceus.cpu())
            accs_ceus_avg.append(accs_ceus)
            miou_ceus_avg.append(miou_ceus)
            mdice_ceus_avg.append(mdice_ceus)

            src_masks_us = outputs["seg_us"]
            target_masks_us = []
            for t in targets:
                target_masks_us.append(t['masks_us'])
            target_masks_us = torch.stack(target_masks_us).squeeze(1)

            accs_us, miou_us, mdice_us, kappas_us = utils.metrics.calculate_metrics(src_masks_us.cpu(),
                                                                                    target_masks_us.cpu())
            accs_us_avg.append(accs_us)
            miou_us_avg.append(miou_us)
            mdice_us_avg.append(mdice_us)

            # preds = torch.argmax(src_masks_ceus, dim=1).float()
            # preds_np = preds.cpu().numpy()[0]
            # label_np = target_masks_ceus.numpy()[0]
            # imgs_list = [preds_np, label_np]
            # titles = ["pre", "label"]
            # title = "label_%s.png: acc: %.3f | iou: %.3f | dice: %.3f" % (val_steps, accs_ceus, miou_ceus, mdice_ceus)
            # for i in range(2):
            #     ax = plt.subplot(1, 2, i + 1)
            #     plt.xticks([]), plt.yticks([])
            #     plt.imshow(imgs_list[i], 'gray')
            #     ax.set_title(titles[i])
            # plt.suptitle(title)
            # plt.savefig(str(output_dir) + '/output_imgs/' + "label_%s.png" % val_steps)
            # val_steps += 1

    labels_metrics = calculate_final_metrics(tns, fps, fns, tps)

    valinfo1 = "Evaluating: acc: %.4f | sens: %.4f | spec: %.4f | ppv: %.4f | npv: %.4f | f1: %.4f" \
                % (labels_metrics["accuracy"],
                   labels_metrics["sensitivity"],
                   labels_metrics["specificity"],
                   labels_metrics["ppv"],
                   labels_metrics["npv"],
                   labels_metrics["f1_score"])
    print(valinfo1)
    valinfo2 = "Evaluating: acc_mask_ceus: %.4f | miou_ceus: %.4f | mdice_ceus: %.4f" \
                % (sum(accs_ceus_avg) / len(accs_ceus_avg), sum(miou_ceus_avg) / len(miou_ceus_avg), sum(mdice_ceus_avg) / len(mdice_ceus_avg))
    print(valinfo2)
    valinfo3 = "Evaluating: acc_mask_us: %.4f | miou_us: %.4f | mdicus: %.4f" \
                % (sum(accs_us_avg) / len(accs_us_avg), sum(miou_us_avg) / len(miou_us_avg), sum(mdice_us_avg) / len(mdice_us_avg))
    print(valinfo3)
    return val_steps

