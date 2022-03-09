import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig

from DST_als.data import arg_parse

gpu_id = '0'
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

args = arg_parse()
model_config = BertConfig.from_json_file(args.bert_config_path)
model_config.dropout = args.dropout
model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
model_config.hidden_dropout_prob = args.hidden_dropout_prob

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, op_ids, n_op):
        op_ids = op_ids.reshape(-1)
        x = x.reshape(-1, 768)

        def del_ele(arr, index):
            return torch.cat((arr[0:index], arr[index + 1:]), dim=0)

        """loss统计"""
        def calculate_loss(cur_op_id):
            cur_losses = []
            cur_label_type_axis = (op_ids == cur_op_id).nonzero()
            cur_label_type_index = cur_label_type_axis.expand(-1, x.size(-1)).to(device)
            cur_label_type_slot_f = torch.gather(x, 0, cur_label_type_index)
            if len(cur_label_type_slot_f) != 0:
                _cur_label_type_slot_f = cur_label_type_slot_f.unsqueeze(0).expand(cur_label_type_axis.size(0), -1, -1)
                cur_label_type_sim = self.cos(_cur_label_type_slot_f.transpose(0, 1),
                                              _cur_label_type_slot_f) / self.temp

                _cur_label_type_slot_f = cur_label_type_slot_f.unsqueeze(0).expand(len(op_ids), -1, -1).transpose(0, 1)
                _x = x.unsqueeze(0).expand(cur_label_type_axis.size(0), -1, -1)
                all_sim = self.cos(_cur_label_type_slot_f, _x) / self.temp

                for i in range(cur_label_type_sim.size(0)):
                    cur_label_type_sim_i = del_ele(cur_label_type_sim[i], i)
                    all_sim_i = del_ele(all_sim[i], cur_label_type_axis[i][0])

                    all_sim_i_lse = torch.logsumexp(all_sim_i, dim=0)
                    loss = (-torch.sum(cur_label_type_sim_i) + len(cur_label_type_sim_i) * all_sim_i_lse) / len(
                        cur_label_type_sim_i)
                    if torch.isnan(loss):
                        cur_losses.append(torch.tensor(0.0, requires_grad=True, device=device))
                    else:
                        cur_losses.append(loss)
                return torch.mean(torch.vstack(cur_losses))
            else:
                return torch.tensor(0.0, device=device)

        losses = []
        for i in range(n_op):
            losses.append(calculate_loss(i))
        return torch.mean(torch.vstack(losses))


class SomDST(BertPreTrainedModel):
    def __init__(self, args, num_labels, update_id=2, n_op=3, n_domain=5, exclude_domain=True):
        super(SomDST, self).__init__(model_config)
        self.hidden_size = model_config.hidden_size
        self.encoder = Encoder(model_config, n_op, n_domain, update_id, exclude_domain)
        self.decoder = Decoder(num_labels)
        # self.apply(self.init_weights)

        self.contrastive_loss = ContrastiveLoss()

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                op_ids=None, value_match_ids=None):

        enc_outputs = self.encoder(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   state_positions=state_positions,
                                   attention_mask=attention_mask,
                                   )

        domain_scores, state_scores, state_output, pooled_output = enc_outputs
        # gen_scores = self.decoder(input_ids, decoder_inputs, sequence_output,
        #                           pooled_output, max_value, teacher)
        if op_ids is None:
            _, op_ids = state_scores.view(-1, 3).max(-1)
            contrastive_loss = 0
        else:
            contrastive_loss = self.contrastive_loss(state_output, op_ids, 3)

        value_match_loss, value_match_pred_ids = self.decoder(state_output, op_ids, value_match_ids)

        return domain_scores, state_scores, value_match_loss, value_match_pred_ids, contrastive_loss


class Encoder(nn.Module):
    def __init__(self, config, n_op, n_domain, update_id, exclude_domain=True):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.exclude_domain = exclude_domain
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.action_cls = nn.Linear(config.hidden_size, n_op)
        # if self.exclude_domain is not True:
        #     self.domain_cls = nn.Linear(config.hidden_size, n_domain)
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                ):
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)
        state_scores = self.action_cls(self.dropout(state_output))  # B,J,4
        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # dummy
        else:
            domain_scores = self.domain_cls(self.dropout(pooled_output))

        # batch_size = state_scores.size(0)
        # if op_ids is None:
        #     op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
        # if max_update is None:
        #     max_update = op_ids.eq(self.update_id).sum(-1).max().item()
        #
        # gathered = []
        # for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update
        #     if a.sum().item() != 0:
        #         v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
        #         n = v.size(1)
        #         gap = max_update - n
        #         if gap > 0:
        #             zeros = torch.zeros(1, 1*gap, self.hidden_size, device=input_ids.device)
        #             v = torch.cat([v, zeros], 1)
        #     else:
        #         v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
        #     gathered.append(v)
        # decoder_inputs = torch.cat(gathered)
        return domain_scores, state_scores, state_output, pooled_output.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, num_labels):
        super(Decoder, self).__init__()
        self.num_labels = num_labels
        self.mlps = []
        for i in range(len(num_labels)):
            self.mlps.append(nn.Linear(768, num_labels[i]))
        self.mlps = nn.ModuleList(self.mlps)
        self.loss_fnc = nn.CrossEntropyLoss()

    def forward(self, state_output, op_ids, value_match_ids):
        losses = []
        value_pred_ids = []

        state_output = state_output.view(-1, 768)
        if value_match_ids is not None:
            value_match_ids = value_match_ids.view(-1)
            for i in range(len(value_match_ids)):
                if value_match_ids[i] != -1:
                    slot_id = i % 30
                    value_score = self.mlps[slot_id](state_output[i, :])
                    loss = self.loss_fnc(torch.unsqueeze(value_score, 0), torch.unsqueeze(value_match_ids[i], 0))
                    losses.append(loss)
                    _, cur_pred_id = value_score.max(-1)
                    value_pred_ids.append(cur_pred_id)
                else:
                    value_pred_ids.append(-1)
        else:
            for i in range(30):
                if op_ids[i] == 2:
                    value_score = self.mlps[i](state_output[i, :])
                    _, cur_pred_id = value_score.max(-1)
                    value_pred_ids.append(cur_pred_id)
                else:
                    value_pred_ids.append(-1)
        if len(losses) != 0:
            loss = losses[0]
            for i in range(1, len(losses)):
                loss += losses[i]

            return loss / len(losses), value_pred_ids

        return torch.tensor(0., requires_grad=True), value_pred_ids
