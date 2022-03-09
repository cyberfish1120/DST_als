import torch

import torch.nn as nn
from tqdm import tqdm

from copy import deepcopy
from torch.utils.data import RandomSampler, DataLoader


class Trainer:
    """

    """

    def __init__(self, args, model, device):
        self.args = args
        self.device = device
        self.model = model.to(device)

        self.optimizer = self.config_optimizer()
        # self.scheduler = self.config_scheduler(self.optimizer)
        self.loss_fnc = self.config_loss()
        # metrics = self.config_metrics()
        # if not isinstance(metrics, list):
        #     metrics = [metrics]
        # self.metrics = metrics

    def fit(self,
            train_data,
            dev_data=None,
            metrics=None,
            callbacks=None,
            ):
        # model.stop_training = False

        # Setup callbacks
        # self.callbacks = callbacks = self.config_callbacks(verbose, epochs, callbacks=callbacks)
        # callbacks.on_train_begin()

        try:
            for epoch in range(self.args.epochs):
                # callbacks.on_epoch_begin(epoch)
                train_logs = self.train_step(train_data)
                print(f'Epoch: {epoch}/{self.args.epochs} train info: {train_logs}')

                if dev_data:
                    valid_logs = self.test_step(dev_data)
                    print(f'Epoch: {epoch}/{self.args.epochs} dev info: {train_logs}')

                # callbacks.on_epoch_end(epoch, logs)

                # if model.stop_training:
                #     print(f"Early Stopping at Epoch {epoch}", file=sys.stderr)
                #     break

        finally:
            # callbacks.on_train_end()
            pass

        return self

    def train_step(self, train_data):
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=self.args.batch_size,
                                      collate_fn=train_data.collate_fn,
                                      num_workers=self.args.num_workers
                                      )
        for epoch in range(self.args.epochs):
            batch_loss = []
            self.model.train()
            for step, batch in tqdm(enumerate(train_dataloader), total=int(len(train_data) / self.args.batch_size) + 1):
                batch = [b.to(self.device) if not isinstance(b, int) else b for b in batch]
                input_ids, input_mask, segment_ids, state_position_ids, op_ids, value_match_ids = batch

                teacher = None

                domain_scores, state_scores, value_match_loss, _, constra_loss = self.model(input_ids=input_ids,
                                                                                       token_type_ids=segment_ids,
                                                                                       state_positions=state_position_ids,
                                                                                       attention_mask=input_mask,
                                                                                       op_ids=op_ids,
                                                                                       value_match_ids=value_match_ids,
                                                                                       )
                loss_s = self.loss_fnc(state_scores.view(-1, 3), op_ids.view(-1))
                # loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
                #                                                     gen_ids.contiguous(),
                #                                                     tokenizer.vocab['[PAD]'])
                # loss = loss_s + loss_g
                if self.args.cl_rate == 0:
                    loss = loss_s + value_match_loss
                else:
                    loss = loss_s + self.args.cl_rate * constra_loss + value_match_loss
                # if args.exclude_domain is not True:
                #     loss_d = loss_fnc(domain_scores.view(-1, len(domain2id)), domain_ids.view(-1))
                #     loss = loss + loss_d
                batch_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()
                # enc_scheduler.step()
                # dec_optimizer.step()
                # dec_scheduler.step()
                self.model.zero_grad()

    def evaluate(self, test_data):
        pass

    def model_evaluation(self, model, test_data, tokenizer, slot_meta, value_labels, epoch, op_code='4',
                         is_gt_op=False, is_gt_p_state=False, is_gt_gen=False):
        model.eval()
        op2id = {'none': 0, 'do not care': 1, 'other': 2}
        id2op = {v: k for k, v in op2id.items()}
        # id2domain = {v: k for k, v in domain2id.items()}

        slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
        final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
        op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
        all_op_F1_count = {k: 0 for k in op2id}

        tp_dic = {k: 0 for k in op2id}
        fn_dic = {k: 0 for k in op2id}
        fp_dic = {k: 0 for k in op2id}

        results = {}
        last_dialog_state = []
        wall_times = []

        op_results = []
        for iii in range(3):
            op_results.append({'S': 1e-10, 'G': 1e-10, 'P': 1e-10})

        for di, i in tqdm(enumerate(test_data), total=len(test_data)):
            if i.turn_id == '0':
                last_dialog_state = ['none'] * len(slot_meta)

            if is_gt_p_state is False:
                i.last_dialog_state = deepcopy(last_dialog_state)
                i.make_instance(tokenizer, word_dropout=0.)
            else:  # ground-truth previous dialogue state
                last_dialog_state = deepcopy(i.gold_p_state)
                i.last_dialog_state = deepcopy(last_dialog_state)
                i.make_instance(tokenizer, word_dropout=0.)

            input_ids = torch.LongTensor([i.input_id]).to(self.device)
            input_mask = torch.FloatTensor([i.input_mask]).to(self.device)
            segment_ids = torch.LongTensor([i.segment_id]).to(self.device)
            state_position_ids = torch.LongTensor([i.slot_position]).to(self.device)
            gold_value_ids = torch.LongTensor([i.value_match_ids]).to(self.device)
            # d_gold_op, _ = make_turn_label(slot_meta, last_dialog_state, i.turn_dialog_state,
            #                                   tokenizer, op_code, dynamic=True)

            # op_labels = [label if label == 'none' or label == 'do not care' else 'other' for label in i.turn_dialog_state]

            d_gold_op = [op2id[a] for a in i.op_labels]
            gold_op_ids = torch.LongTensor([d_gold_op]).to(self.device)

            MAX_LENGTH = 9
            with torch.no_grad():
                # ground-truth state operation
                gold_op_inputs = gold_op_ids if is_gt_op else None
                d, s, _, value_ids, _ = model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              state_positions=state_position_ids,
                                              attention_mask=input_mask,
                                              op_ids=gold_op_inputs,
                                              value_match_ids=None
                                              )

            _, op_ids = s.view(-1, len(op2id)).max(-1)

            # if g.size(1) > 0:
            #     generated = g.squeeze(0).max(-1)[1].tolist()
            # else:
            #     generated = []

            # if is_gt_op:
            #     pred_ops = [id2op[a] for a in gold_op_ids[0].tolist()]
            # else:
            #     pred_ops = [id2op[a] for a in op_ids.tolist()]
            # gold_ops = [id2op[a] for a in d_gold_op]

            # if is_gt_gen:
            #     # ground_truth generation
            #     gold_gen = {'-'.join(ii.split('-')[:2]): ii.split('-')[-1] for ii in i.turn_dialog_state}
            # else:
            #     gold_gen = {}
            # generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
            #                                               generated, tokenizer, op_code)
            """TODO:生成状态"""
            last_dialog_state = []
            for op_i, id in enumerate(op_ids):
                if id == 0:
                    last_dialog_state.append('none')
                elif id == 1:
                    last_dialog_state.append('do not care')
                else:
                    last_dialog_state.append(value_labels[op_i][value_ids[op_i]])

            pre_state = ' '.join(last_dialog_state)
            gold_state = ' '.join(i.turn_dialog_state)
            if set(pre_state) == set(gold_state):
                joint_acc += 1

            # key = str(i.id) + '_' + str(i.turn_id)
            # results[key] = [last_dialog_state, i.turn_dialog_state]

            # Compute prediction slot accuracy
            # temp_acc = compute_acc(set(i.turn_dialog_state), set(last_dialog_state), slot_meta)
            # slot_turn_acc += temp_acc

            # Compute prediction F1 score
            # temp_f1, temp_r, temp_p, count = compute_prf(i.turn_dialog_state, last_dialog_state)
            # slot_F1_pred += temp_f1
            # slot_F1_count += count

            # Compute operation accuracy
            # temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_ops, gold_ops)]) / len(pred_ops)
            # op_acc += temp_acc

            # if True:
            #     final_count += 1
            #     if set(last_dialog_state) == set(i.turn_dialog_state):
            #         final_joint_acc += 1
            #     final_slot_F1_pred += temp_f1
            #     final_slot_F1_count += count

            def clf_eval(clf_i, op_results, gold, pred):
                op_results[clf_i]['S'] += ((gold == clf_i) & (pred == clf_i)).sum()
                op_results[clf_i]['G'] += (gold == clf_i).sum()
                op_results[clf_i]['P'] += (pred == clf_i).sum()

                return op_results

            for iii in range(3):
                op_results = clf_eval(iii, op_results, torch.squeeze(gold_op_ids), op_ids)

            # if di % 2000 == 0:
            #     print(f'{di}/{len(test_data)}')

        #     # Compute operation F1 score
        #     for p, g in zip(pred_ops, gold_ops):
        #         all_op_F1_count[g] += 1
        #         if p == g:
        #             tp_dic[g] += 1
        #             op_F1_count[g] += 1
        #         else:
        #             fn_dic[g] += 1
        #             fp_dic[p] += 1
        #
        # joint_acc_score = joint_acc / len(test_data)
        # turn_acc_score = slot_turn_acc / len(test_data)
        # slot_F1_score = slot_F1_pred / slot_F1_count
        # op_acc_score = op_acc / len(test_data)
        # final_joint_acc_score = final_joint_acc / final_count
        # final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
        # latency = np.mean(wall_times) * 1000
        # op_F1_score = {}
        # for k in op2id.keys():
        #     tp = tp_dic[k]
        #     fn = fn_dic[k]
        #     fp = fp_dic[k]
        #     precision = tp / (tp+fp) if (tp+fp) != 0 else 0
        #     recall = tp / (tp+fn) if (tp+fn) != 0 else 0
        #     F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        #     op_F1_score[k] = F1
        #
        # print("------------------------------")
        # print('op_code: %s, is_gt_op: %s, is_gt_p_state: %s, is_gt_gen: %s' % \
        #       (op_code, str(is_gt_op), str(is_gt_p_state), str(is_gt_gen)))
        # print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
        # print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
        # print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
        # print("Epoch %d op accuracy : " % epoch, op_acc_score)
        # print("Epoch %d op F1 : " % epoch, op_F1_score)
        # print("Epoch %d op hit count : " % epoch, op_F1_count)
        # print("Epoch %d op all count : " % epoch, all_op_F1_count)
        # print("Final Joint Accuracy : ", final_joint_acc_score)
        # print("Final slot turn F1 : ", final_slot_F1_score)
        # print("Latency Per Prediction : %f ms" % latency)
        # print("-----------------------------\n")
        # # json.dump(results, open('preds_%d.json' % epoch, 'w'))
        # per_domain_join_accuracy(results, slot_meta)
        #
        # scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
        #           'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score,
        #           'op_acc': op_acc_score, 'op_f1': op_F1_score, 'final_slot_f1': final_slot_F1_score}
        # return scores
        for iii in range(len(op2id)):
            op_results[iii]['pre'] = op_results[iii]['S'] / op_results[iii]['P']
            op_results[iii]['rec'] = op_results[iii]['S'] / op_results[iii]['G']
            op_results[iii]['f1'] = 2 * op_results[iii]['S'] / (op_results[iii]['P'] + op_results[iii]['G'])

        results['joint_acc'] = joint_acc / len(test_data)
        results['none_F1'] = float(op_results[0]['f1'])
        results['none_Pre'] = float(op_results[0]['pre'])
        results['none_Rec'] = float(op_results[0]['rec'])
        results['dnc_F1'] = float(op_results[1]['f1'])
        results['dnc_Pre'] = float(op_results[1]['pre'])
        results['dnc_Rec'] = float(op_results[1]['rec'])
        results['other_F1'] = float(op_results[2]['f1'])
        results['other_Pre'] = float(op_results[2]['pre'])
        results['other_Rec'] = float(op_results[2]['rec'])

        print('epoch:%d\nnone_f1:%.4f\tdnc_f1:%.4f\tother:%.4f\tjoint_acc:%.4f' % (epoch,
                                                                                   op_results[0]['f1'],
                                                                                   op_results[1]['f1'],
                                                                                   op_results[2]['f1'],
                                                                                   joint_acc / len(test_data)))
        return results


    def config_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.enc_lr)

    def config_loss(self):
        return nn.CrossEntropyLoss()