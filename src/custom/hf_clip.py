from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass
import torch.nn.functional as F
import torch
from torch import nn


from transformers.models.clip.modeling_clip import CLIPModel, CLIPPreTrainedModel, CLIPOutput, CLIPConfig
from transformers import AutoModel, AutoProcessor, AutoTokenizer

class HF_CLIPScoreFusion(CLIPModel):
    config_class = CLIPConfig
    def __init__(self, config:CLIPConfig):
        super().__init__(config)
    
    def fuse_embeddings(self, text_embeds, image_embeds):
        return image_embeds + text_embeds
    
    def encode_multimodal_input(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        
        return self.fuse_embeddings(text_embeds, image_embeds)
        
    def compute_inbatch_contrastive_loss(self, batch):
        """
         adapted from the CLIP codebase and UniVL-DR codebase

        :param model:
        :param batch:
        :param loss_function:
        :return:
        """
        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask_batched = batch["txt_mask_batched"]
        image_mask_batched = batch["image_mask_batched"]
        index_mapping = batch["index_mapping"]
        enable_hard_neg = "neg_cand_list" in index_mapping

        # Compute embeddings
        embeddings = self.encode_multimodal_input(txt_batched, image_batched, txt_mask_batched, image_mask_batched)

        # Extract embeddings
        q_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]  # shape: [bs, embed_dim]
        p_embeds = embeddings[torch.tensor(index_mapping["pos_cand"]).flatten()]  # shape: [bs, embed_dim]
        n_embeds = None
        if enable_hard_neg:
            n_embeds = embeddings[torch.tensor(index_mapping["neg_cand_list"])]  # [bs, neg_num, embed_dim]
        bs = q_embeds.size(0)

        # Normalized features
        q_embeds = F.normalize(q_embeds, dim=-1)
        p_embeds = F.normalize(p_embeds, dim=-1)

        logit_scale = self.get_logit_scale()

        # We gather tensors from all gpus
        if self.gather_embeddings:
            all_p_embeds = torch.cat(torch.distributed.nn.all_gather(p_embeds), dim=0)  # [bs * num_gpus, embed_dim]

        if enable_hard_neg:
            # Normalize the negative embeddings
            n_embeds = F.normalize(n_embeds, dim=-1)

            # Number of in-batch positives to add as negatives
            in_batch_neg_num = min(bs - 1, self.in_batch_neg_num)

            # Augment neg_cand_embeddings with a subset of in-batch positive candidates from other queries
            mask = torch.eye(bs).to(n_embeds.device) == 0
            in_batch_negs = p_embeds.unsqueeze(1).expand(-1, bs, -1)[mask].reshape(bs, bs - 1, -1)
            in_batch_negs = in_batch_negs[:, :in_batch_neg_num, :]
            aug_n_embeds = torch.cat([n_embeds, in_batch_negs], dim=1)  # [bs, neg_num + in_batch_neg_num, embed_dim]

            # Compute similarity scores for positives and negatives
            pos_scores = (q_embeds * p_embeds).sum(-1) * logit_scale  # [bs]
            neg_scores = (q_embeds.unsqueeze(1) * aug_n_embeds).sum(-1) * logit_scale  # [bs, neg_num +in_batch_neg_num]
            logit_matrix = torch.cat([pos_scores.unsqueeze(-1), neg_scores], 1)  # [bs, neg_num + in_batch_neg_num + 1]

            # Compute log softmax over the matrix
            lsm = F.log_softmax(logit_matrix, dim=1)

            # The NNL loss for the positive candidate
            loss = torch.mean(-1.0 * lsm[:, 0])

            # Compute accuracy by checking which instances have the positive candidate as the most similar one
            _max_score, max_idxs = torch.max(logit_matrix, 1)
            accuracy = (max_idxs == 0).sum() / bs
        else:
            if self.gather_embeddings:
                score = torch.matmul(q_embeds, all_p_embeds.t()) * logit_scale  # [bs, bs * num_gpus]
                gpu_id = torch.distributed.get_rank()
                sim_targets = (gpu_id * bs + torch.arange(bs)).to(score.device)  # [bs]
            else:
                score = torch.matmul(q_embeds, p_embeds.t()) * logit_scale  # [bs, bs]
                sim_targets = torch.arange(bs).to(score.device)  # [bs]

            # compute loss
            loss = self.loss_function(score, sim_targets)
            _max_score, max_idxs = torch.max(score, 1)
            accuracy = (max_idxs == sim_targets).sum() / bs

        outputs = {"loss": loss, "accuracy": accuracy}
        return outputs
    
    def forward(self, batch, encode_mbeir_batch=False):
        if encode_mbeir_batch:
            return self.encode_mbeir_batch(batch)
        return self.compute_inbatch_contrastive_loss(batch)
    
    def encode_mbeir_batch(self, batch):
        # Get hashed id_list
        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        # Compute embeddings
        embeddings = self.encode_multimodal_input(
            batch["txt_batched"], batch["image_batched"], batch["txt_mask_batched"], batch["image_mask_batched"]
        )
        assert embeddings.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
        return embeddings, id_list
