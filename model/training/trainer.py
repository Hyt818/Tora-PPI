'''
Purpose: To implement the main training and validation cycle of Tora-PPI, responsible for forward propagation, loss calculation, backpropagation, metric calculation, and checkpoint saving.
Inputs: Model, PPI graph, merged residue graph features, sequence features, loss, optimizer, scheduler, and training parameters.
Outputs: Training log, TensorBoard scalar, train/best/final checkpoint.
'''

import math
import os
import random

import torch
import torch.nn as nn

from model.training.losses import info_nce_loss, vicreg_loss
from model.utils.utils import Metrictor_PPI, print_file

def train(
    batch,
    p_x_all,
    p_edge_all,
    p_edge_arr,
    p_seq_all,
    model,
    graph,
    ppi_list,
    loss_fn,
    optimizer,
    device,
    result_file_path,
    summary_writer,
    save_path,
    batch_size=11000,
    epochs=500,
    scheduler=None,
    enable_early_stop=False,
    early_stop_lr=1e-6,
):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    stop_training = False

    prop_edge_index = graph.edge_index_got

    for epoch in range(epochs):
        if stop_training:
            break

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)

        for step in range(steps):
            if step == steps - 1:
                train_edge_id = graph.train_mask[step * batch_size:]
            else:
                train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            pred_edge_index = graph.edge_index[:, train_edge_id]

            output, z_gnn, z_seq, g, s = model(
                batch,
                p_x_all,
                p_edge_all,
                p_edge_arr,
                p_seq_all,
                prop_edge_index,
                pred_edge_index,
                graph,
                return_align=True
            )

            label = graph.edge_attr_1[train_edge_id]
            label = label.type(torch.FloatTensor).to(device)

            ppi_loss = loss_fn(output, label)

            lambda1 = 0.4
            tau = 0.01
            align_loss = info_nce_loss(z_gnn, z_seq, tau=tau)

            lambda2 = 0.4
            graph_loss = vicreg_loss(g, s)

            loss = ppi_loss + lambda1 * align_loss + lambda2 * graph_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            prob = m(output)
            pre_result = (prob > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(
                pre_result.cpu().data,
                label.cpu().data,
                prob.cpu().data
            )
            metrics.show_result(is_print=False)

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            summary_writer.add_scalar('train/ppi_loss', ppi_loss.item(), global_step)
            summary_writer.add_scalar('train/info_nce_loss', align_loss.item(), global_step)
            summary_writer.add_scalar('train/vicreg_loss', graph_loss.item(), global_step)
            summary_writer.add_scalar('train/total_loss', loss.item(), global_step)

            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1

            print_file(
                'epoch: {}, step: {}, Train: total_loss: {}, precision: {}, recall: {}, f1: {}'.format(
                    epoch,
                    step,
                    loss.item(),
                    metrics.Precision,
                    metrics.Recall,
                    metrics.F1
                ),
                save_file_path=result_file_path
            )

        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict()
            },
            os.path.join(save_path, 'gnn_model_train.ckpt')
        )

        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                pred_edge_index = graph.edge_index[:, valid_edge_id]

                output = model(
                    batch,
                    p_x_all,
                    p_edge_all,
                    p_edge_arr,
                    p_seq_all,
                    prop_edge_index,
                    pred_edge_index,
                    graph,
                    return_align=False
                )

                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                prob = m(output)
                pre_result = (prob > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(prob.cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim=0)

        metrics = Metrictor_PPI(
            valid_pre_result_list,
            valid_label_list,
            true_prob_list
        )
        metrics.show_result(is_print=False)

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        train_loss = loss_sum / steps
        valid_loss = valid_loss_sum / valid_steps

        if scheduler is not None and enable_early_stop:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            if current_lr < early_stop_lr:
                stop_training = True

        if scheduler is not None:
            scheduler.step(train_loss)

            print_file(
                'epoch: {}, now learning rate: {}'.format(
                    epoch,
                    scheduler.optimizer.param_groups[0]['lr']
                ),
                save_file_path=result_file_path
            )

            if stop_training:
                print_file(
                    'Learning rate dropped below {}. Final validation results:'.format(early_stop_lr),
                    save_file_path=result_file_path
                )

                torch.save(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict()
                    },
                    os.path.join(save_path, 'gnn_model_final.ckpt')
                )
                break

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict()
                },
                os.path.join(save_path, 'gnn_model_valid_best.ckpt')
            )

        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)

        print_file(
            'epoch: {}, Training_avg: loss: {}, recall: {}, precision: {}, F1: {}, '
            'Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, '
            'Best valid_f1: {}, in {} epoch'.format(
                epoch,
                train_loss,
                recall,
                precision,
                f1,
                valid_loss,
                metrics.Recall,
                metrics.Precision,
                metrics.F1,
                global_best_valid_f1,
                global_best_valid_f1_epoch
            ),
            save_file_path=result_file_path
        )

