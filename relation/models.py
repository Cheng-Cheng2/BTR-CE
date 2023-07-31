import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import AlbertModel, AlbertPreTrainedModel
from transformers import AutoModel, AutoConfig
import geoopt as gt

from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
import torch.nn.functional as F
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

BertLayerNorm = torch.nn.LayerNorm
class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0] # # torch.Size([8, 256, 768]): [B, seq_L, H]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)]) #unsqueeze: returns a new tensor with a dimension of size one inserted at the specified position.
        # a[i].unsqueeze(0).shape: torch.Size([1, 768])
        # sub_output.shape: [8, 768])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1) # torch.Size([8, 1536])
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep) # B x H*2 x self.num_labels

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class OtherLLMForRelation(nn.Module):
    def __init__(self, config, num_rel_labels):
        #super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = model#BertModel(config)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0] # # torch.Size([8, 256, 768]): [B, seq_L, H]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)]) #unsqueeze: returns a new tensor with a dimension of size one inserted at the specified position.
        # a[i].unsqueeze(0).shape: torch.Size([1, 768])
        # sub_output.shape: [8, 768])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1) # torch.Size([8, 1536])
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep) # B x H*2 x self.num_labels

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits   

class HyperGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, ball):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ball = ball

        k = (1 / self.hidden_size) ** 0.5
        self.w_z = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.hidden_size).uniform_(-k, k)
        )
        self.w_r = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.hidden_size).uniform_(-k, k)
        )
        self.w_h = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.hidden_size).uniform_(-k, k)
        )
        self.u_z = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.input_size).uniform_(-k, k)
        )
        self.u_r = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.input_size).uniform_(-k, k)
        )
        self.u_h = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.input_size).uniform_(-k, k)
        )
        self.b_z = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )
        self.b_r = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )
        self.b_h = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )

    def transition(self, W, h, U, x, hyp_b):
        W_otimes_h = self.ball.mobius_matvec(W, h)
        U_otimes_x = self.ball.mobius_matvec(U, x)
        Wh_plus_Ux = self.ball.mobius_add(W_otimes_h, U_otimes_x)

        return self.ball.mobius_add(Wh_plus_Ux, hyp_b)

    def forward(self, hyp_x, hidden):
        z = self.transition(self.w_z, hidden, self.u_z, hyp_x, self.b_z)
        z = torch.sigmoid(self.ball.logmap0(z))

        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = torch.sigmoid(self.ball.logmap0(r))

        r_point_h = self.ball.mobius_pointwise_mul(hidden, r)
        h_tilde = self.transition(self.w_h, r_point_h, self.u_h, hyp_x, self.b_h)

        minus_h_oplus_htilde = self.ball.mobius_add(-hidden, h_tilde)
        new_h = self.ball.mobius_add(
            hidden, self.ball.mobius_pointwise_mul(minus_h_oplus_htilde, z)
        )

        return new_h

class HyperGRU(nn.Module):
    def __init__(self, input_size, hidden_size, ball):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ball = ball
        #self.default_dtype = default_dtype
        self.gru_cell = HyperGRUCell(self.input_size, self.hidden_size, ball=self.ball)

    def reset_parameters(self):
        self.gru_cell.reset_parameters()

    def init_gru_state(self, batch_size, hidden_size, cuda_device):
        # return torch.zeros(
        #     (batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device
        # )
        return torch.zeros(
            (batch_size, hidden_size))

    def forward(self, inputs):
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return torch.stack(outputs).transpose(0, 1)

class EventTempRel_HGRU(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(EventTempRel_HGRU, self).__init__(config)
        self.bert = BertModel(config)
        self.dim_in = 768
        self.dim_hidden = 128
        self.dim_out = 64
        self.bigramStats_dim=1
        self.non_lin='id'
        self.granularity=0.05
        self.common_sense_emb_dim=64
        self.dropout=0.1
        self.num_labels = num_rel_labels

        #self.dim_in = self.dim_in
        #self.dim_out = sdim_out
        #self.dim_hidden = dim_hidden
        #self.non_lin = non_lin
        #self.device = device
        #self.dtype = dtype

        #self.model_name = model_name
        #self.bert = AutoModel.from_pretrained(self.model_name, return_dict=True)
        #self.bert = BertPreTrainedModel(cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), config = AutoConfig.from_pretrained('bert-base-uncased'), num_rel_labels=num_rel_labels)
        #self.config = AutoConfig.from_pretrained('bert-base-uncased')
        #self.bert = BertModel(config)

        #self.bert.train()

        #self.bigramStats_dim = bigramStats
        #self.granularity = granularity
        #self.common_sense_emb_dim = common_sense_emb_dim
        #self.common_sense_emb = nn.Embedding(int(1.0/self.granularity)*self.bigramStats_dim,self.common_sense_emb_dim)

        self.ball = gt.Stereographic(-1) # declare a Poincare model using Geoopt's Stereographic class. The curvature is -1

        if self.dropout > 0.:
            self.dropout = nn.Dropout(p=self.dropout)
        else:
            self.dropout = None
        
        self.HyperGRU = HyperGRU(input_size=self.dim_in, hidden_size=self.dim_hidden,
                                ball=self.ball)

        self.W_ff_u = nn.Parameter(data=torch.zeros(self.dim_out, self.dim_hidden))
        self.W_ff_v = nn.Parameter(data=torch.zeros(self.dim_out, self.dim_hidden))
        nn.init.uniform_(self.W_ff_u, -1.0/(self.dim_hidden+self.dim_out), 1.0/(self.dim_hidden+self.dim_out))
        nn.init.uniform_(self.W_ff_v, -1.0/(self.dim_hidden+self.dim_out), 1.0/(self.dim_hidden+self.dim_out))

        self.b_ff = nn.Parameter(data=torch.zeros(self.dim_out))    # zero initialize
        self.b_ff_d = nn.Parameter(data=torch.zeros(self.dim_out))

        self.p_mlr = nn.Parameter(data=torch.zeros(self.num_labels, self.dim_out))    # should these be hyperbolic parameters?
        self.a_mlr = nn.Parameter(data=torch.zeros(self.num_labels, self.dim_out))    # these parameters are on tangent space
        nn.init.uniform_(self.a_mlr, -1.0/self.dim_out, 1.0/self.dim_out)
        nn.init.uniform_(self.p_mlr, -1.0/self.dim_out, 1.0/self.dim_out)

       #self.W_ff_common = nn.Parameter(data=torch.zeros(self.dim_out, self.bigramStats_dim*self.common_sense_emb_dim))
       # nn.init.uniform_(self.W_ff_common, -1.0/(self.dim_out+self.bigramStats_dim*self.common_sense_emb_dim), 1.0/(self.dim_out+self.bigramStats_dim*self.common_sense_emb_dim))

        self.loss = nn.CrossEntropyLoss(reduction='none')
        #self.hyper_para = [self.p_mlr, self.b_ff, self.b_ff_d, self.common_sense_emb.weight, self.W_ff_u, self.W_ff_v] + list(self.HyperGRU.parameters())
        self.hyper_para = [self.p_mlr, self.b_ff, self.b_ff_d, self.W_ff_u, self.W_ff_v] + list(self.HyperGRU.parameters())

        self.euclid_para = [self.a_mlr]
        #self.plm_para = self.bert.parameters()

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
    
    #forward(self, input_ids, s_a_mask, mask1, mask2, common_ids,  token_type_ids=None, attention_mask=None):
        # common_ids [batch, 2]
        #common_emb = self.common_sense_emb(common_ids) # [batch,2,common_dim]
       # common_emb = common_emb.view(common_emb.size(0),-1)

        # sequence [batch, len, embeddings]
        #encoded = self.bert(sequence, s_a_mask).last_hidden_state
        # debug: this is bug
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        #encoded = self.bert
        encoded = outputs[0]
        projected = self.ball.expmap0(encoded)
        hidden = self.HyperGRU(projected)
        
        # u = hidden * mask1 # boardcast
        # v = hidden * mask2
        
        # u = torch.sum(u, dim=1)
        # v = torch.sum(v, dim=1)
        u = torch.cat([a[i].unsqueeze(0) for a, i in zip(hidden, sub_idx)]) #unsqueeze: returns a new tensor with a dimension of size one inserted at the specified position.
        v = torch.cat([a[i].unsqueeze(0) for a, i in zip(hidden, obj_idx)])

        dsq = self.ball.dist(u, v).unsqueeze(1)

        # fully connected layers (concatenation)
        ffnn_u = self.ball.mobius_matvec(self.W_ff_u, u)
        ffnn_v = self.ball.mobius_matvec(self.W_ff_v, v)
        output_ffnn = self.ball.mobius_add(ffnn_u, ffnn_v)
        output_ffnn = self.ball.mobius_add(output_ffnn, self.b_ff)

        # extra feature: distance between u and v
        output_ffnn = self.ball.mobius_add(output_ffnn, self.ball.mobius_scalar_mul(dsq, self.b_ff_d))

        # CSE
        #output_ffnn = self.ball.mobius_add(output_ffnn, self.ball.mobius_matvec(self.W_ff_common, common_emb))

        # non-linear
        output_ffnn = self._non_lin(self.ball.logmap0(output_ffnn))

        if self.dropout:
            output_ffnn = self.dropout(output_ffnn)

        output_ffnn = self.ball.expmap0(output_ffnn)
        logits = self._compute_mlr_logits(output_ffnn)

        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits

    def _non_lin(self, vector):
        if self.non_lin == 'id':
            return vector
        elif self.non_lin == 'relu':
            return torch.nn.functional.relu(vector)
        elif self.non_lin == 'tanh':
            return torch.nn.functional.tanh(vector)
        elif self.non_lin == 'sigmoid':
            return torch.nn.functional.sigmoid(vector)
    
    def _compute_mlr_logits(self, output_before):
        logits = []
        for cl in range(self.num_labels):
            minus_p_plus_x = self.ball.mobius_add(-self.p_mlr[cl], output_before)    # [batch, hidden]
            norm_a = torch.norm(self.a_mlr[cl])
            lambda_px = self._lambda(minus_p_plus_x)    # [batch, 1]
            px_dot_a = torch.sum(minus_p_plus_x * nn.functional.normalize(self.a_mlr[cl].unsqueeze(0), p=2), dim=1)   # [batch, 1]
            logit = 2. * norm_a * torch.asinh(px_dot_a * lambda_px)
            logits.append(logit)
        
        logits = torch.stack(logits, axis=1)
        return logits
    
    @staticmethod
    def _lambda(vector):
        return 2. / (1-torch.sum(vector * vector, dim=1))

# class BertForRelation(BertPreTrainedModel):
#     def __init__(self, config, num_rel_labels):
#         super(BertForRelation, self).__init__(config)
#         self.num_labels = num_rel_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.layer_norm = BertLayerNorm(config.hidden_size * 2)
#         self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
#         self.init_weights()

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
#         outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
#         sequence_output = outputs[0] # # torch.Size([8, 256, 768]): [B, seq_L, H]
#         sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)]) #unsqueeze: returns a new tensor with a dimension of size one inserted at the specified position.
#         # a[i].unsqueeze(0).shape: torch.Size([1, 768])
#         # sub_output.shape: [8, 768])
#         obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
#         rep = torch.cat((sub_output, obj_output), dim=1) # torch.Size([8, 1536])
#         rep = self.layer_norm(rep)
#         rep = self.dropout(rep)
#         logits = self.classifier(rep) # B x H*2 x self.num_labels

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits
        
        
class PlainBertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0] # # torch.Size([8, 256, 768]): [B, seq_L, H]
        #sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)]) #unsqueeze: returns a new tensor with a dimension of size one inserted at the specified position.
        # a[i].unsqueeze(0).shape: torch.Size([1, 768])
        # sub_output.shape: [8, 768])
        #obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        #rep = torch.cat((sub_output, obj_output), dim=1) # torch.Size([8, 1536])
        sequence_output = sequence_output.view(sequence_output.shape[0], -1) #DEBUG: to [B, seq_LxH]
        rep = self.layer_norm(sequence_output)
        rep = self.dropout(rep)
        logits = self.classifier(rep) # B x H*2 x self.num_labels

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class AlbertForRelation(AlbertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None):
        outputs = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False)
        sequence_output = outputs[0]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForRelationApprox(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelationApprox, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_obj_ids=None, sub_obj_masks=None, input_position=None):
        """
        attention_mask: [batch_size, from_seq_length, to_seq_length]
        """
        batch_size = input_ids.size(0)
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]

        sub_ids = sub_obj_ids[:, :, 0].view(batch_size, -1)
        sub_embeddings = batched_index_select(sequence_output, sub_ids)
        obj_ids = sub_obj_ids[:, :, 1].view(batch_size, -1)
        obj_embeddings = batched_index_select(sequence_output, obj_ids)
        rep = torch.cat((sub_embeddings, obj_embeddings), dim=-1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = (sub_obj_masks.view(-1) == 1)
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits

class AlbertForRelationApprox(BertPreTrainedModel):
    """
    ALBERT approximation model is not supported by the current implementation, 
    as Huggingface's Transformers ALBERT doesn't support an attention mask with a shape of [batch_size, from_seq_length, to_seq_length]."
    """
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelationApprox, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_obj_ids=None, sub_obj_masks=None, input_position=None):
        batch_size = input_ids.size(0)
        outputs = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]

        sub_ids = sub_obj_ids[:, :, 0].view(batch_size, -1)
        sub_embeddings = batched_index_select(sequence_output, sub_ids)
        obj_ids = sub_obj_ids[:, :, 1].view(batch_size, -1)
        obj_embeddings = batched_index_select(sequence_output, obj_ids)
        rep = torch.cat((sub_embeddings, obj_embeddings), dim=-1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = (sub_obj_masks.view(-1) == 1)
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits
