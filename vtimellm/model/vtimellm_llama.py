import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM, LlamaModel, LlamaForCausalLM
from vtimellm.model.llama_config import LlamaConfig
from vtimellm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, SEG_START, SEG_END
from transformers.modeling_outputs import CausalLMOutputWithPast
from .vtimellm_arch import VTimeLLMMetaModel, VTimeLLMMetaForCausalLM

def giou_1d_loss(prediction, groundtruth):
    # Ensure the start is always less than or equal to the end
    pred_start, pred_end = torch.min(prediction, dim=1)[0], torch.max(prediction, dim=1)[0]
    gt_start, gt_end = torch.min(groundtruth, dim=1)[0], torch.max(groundtruth, dim=1)[0]
    
    # Intersection: Maximum of the start points, minimum of the end points
    intersection_start = torch.max(pred_start, gt_start)
    intersection_end = torch.min(pred_end, gt_end)
    intersection = torch.clamp(intersection_end - intersection_start, min=0)
    
    # Union: Sum of individual lengths minus the intersection
    pred_length = pred_end - pred_start
    gt_length = gt_end - gt_start
    union = pred_length + gt_length - intersection
    
    # Enclosing segment: Smallest segment that can enclose both predicted and ground truth segments
    enclosing_start = torch.min(pred_start, gt_start)
    enclosing_end = torch.max(pred_end, gt_end)
    enclosing_length = enclosing_end - enclosing_start
    
    # IoU for 1D segments
    iou = intersection / union
    
    # GIoU for 1D segments
    giou = iou - (enclosing_length - union) / enclosing_length
    
    # GIoU Loss
    giou_loss = 1 - giou
    
    # Return the mean GIoU loss across all segments
    return giou_loss.mean()


class VTimeLLMConfig(LlamaConfig):
    model_type = "VTimeLLM"

class VTimeLLMLlamaModel(LlamaModel, VTimeLLMMetaModel):
    config_class = VTimeLLMConfig

    def __init__(self, config: LlamaConfig, model_args: dict):
        super(VTimeLLMLlamaModel, self).__init__(config)
        self.model_args = model_args


class VTimeLLMLlamaForCausalLM(LlamaForCausalLM, VTimeLLMMetaForCausalLM):
    config_class = VTimeLLMConfig

    def __init__(self, config, model_args):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VTimeLLMLlamaModel(config, model_args)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.get_model().model_args.segment_head is not None:
            if self.get_model().model_args.segment_head == "linear":
                self.segment_head = nn.Linear(config.hidden_size, 1)
            elif self.get_model().model_args.segment_head == "mlp":
                hidden_dim = config.hidden_size // 4
                self.segment_head_layer1 = nn.Linear(config.hidden_size, hidden_dim)
                self.segment_head_layer2 = nn.Linear(hidden_dim, 1)
            elif self.get_model().model_args.segment_head == "linear_2output":
                self.segment_head = nn.Linear(config.hidden_size, 2)
            elif self.get_model().model_args.segment_head == "mlp_2output":
                hidden_dim = config.hidden_size // 4
                self.segment_head_layer1 = nn.Linear(config.hidden_size, hidden_dim)
                self.segment_head_layer2 = nn.Linear(hidden_dim, 2)


        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        segments: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        # is_generate will skip the prepare_inputs_labels_for_multimodal during evaluation with temporal loss
        is_generate: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        model_args = self.get_model().model_args
        old_input_ids = input_ids.clone()

        if inputs_embeds is None and not is_generate:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return_dict = True
        output_hidden_states = True
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not self.training:
            return outputs
        if (model_args.model_segment_format is None) or (model_args.model_segment_format == "v10"):
            if model_args.segment_head is None:
                return outputs
            else:
                if model_args.model_segment_format == "v10":
                    # for the offset model, use OFFSET as a marker to look for start and end
                    if model_args.predict_goal == "offset":
                        mark_word_id = 32003
                    else:
                        raise NotImplementedError("model_args.predict_goal not implemented")
                else:
                    # other use SEG_START
                    mark_word_id = 32000
                last_hidden_states = outputs.hidden_states[-1]
                num_images_minus_one = (last_hidden_states.shape[1] - old_input_ids.shape[1])
                selected_features = []
                for bs in range(last_hidden_states.shape[0]):
                    if model_args.predict_goal == "offset":
                        offset_indices = torch.where(old_input_ids[bs] == mark_word_id)[0].tolist()
                        offset_indices = list(map(lambda x: x + num_images_minus_one, offset_indices))
    
                        indices = offset_indices
                    elif model_args.predict_goal == "rethink":
                        # TODO: need to check if the difference is 8
                        offset_indices = torch.where(old_input_ids[bs] == mark_word_id)[0].tolist()

                        offset_indices = list(map(lambda x: x - model_args.gap_from_offset + num_images_minus_one, offset_indices))
                        indices = offset_indices
                        indices.sort()
                    else:
                        raise NotImplementedError("model_args.predict_goal not implemented")
                    selected_batch_features = last_hidden_states[bs, indices, :]
                    selected_features.append(selected_batch_features)
                        # seg_start_indices = list(map(lambda x: x - 3 + num_images_minus_one, offset_indices))
                        # seg_end_indices = list(map(lambda x: x - 3 + num_images_minus_one, offset_indices))
                        # indices = seg_start_indices + seg_end_indices
                        # indices.sort()
                selected_features = torch.cat(selected_features)
                if self.model.model_args.segment_head == "linear_2output":
                    segments_predictions = self.segment_head(selected_features).flatten()
                elif self.model.model_args.segment_head == "mlp_2output":
                    segments_predictions = self.segment_head_layer1(selected_features)
                    segments_predictions = torch.relu(segments_predictions)
                    segments_predictions = self.segment_head_layer2(segments_predictions).flatten()
                # elif self.model.model_args.segment_head == "linear":
                #     segments_predictions = self.segment_head(selected_features).flatten()
                # elif self.model.model_args.segment_head == "mlp":
                #     segments_predictions = self.segment_head_layer1(selected_features)
                #     segments_predictions = torch.relu(segments_predictions)
                #     segments_predictions = self.segment_head_layer2(segments_predictions).flatten()
                else:
                    raise NotImplementedError("segment head not implemented")
                segments_predictions = torch.sigmoid(segments_predictions)
                gt = []

                for bs in range(inputs_embeds.shape[0]):
                    # Filter out the -1 elements from the i-th batch
                    valid_elements = segments[bs][segments[bs] != -1]

                    gt.extend(valid_elements)

                assert len(segments_predictions) == len(gt), f"predicted segments should have the same length as groundtruth segments, now is {len(segments_predictions)} and {len(gt)}"
                num_segments = len(gt)
                gt = torch.tensor(gt).to(segments_predictions.device)

                if model_args.loss_type == "vanilla":
                    loss = outputs.loss
                elif model_args.loss_type == "l1_loss":
                    l1_loss = nn.L1Loss()(segments_predictions, gt)
                    normalized_l1_loss = l1_loss / num_segments
                    l1_loss = normalized_l1_loss * self.model.model_args.loss_weight
                    loss = outputs.loss + l1_loss 
                elif model_args.loss_type == "l2_loss":
                    l2_loss = nn.MSELoss()(segments_predictions, gt)
                    normalized_l2_loss = l2_loss / num_segments
                    l2_loss = normalized_l2_loss * self.model.model_args.loss_weight
                    loss = outputs.loss + l2_loss 
                elif model_args.loss_type == "l1_giou_loss":
                    l1_loss = nn.L1Loss()(segments_predictions, gt)
                    normalized_l1_loss = l1_loss / num_segments
                    l1_loss = normalized_l1_loss * self.model.model_args.loss_weight
                    segments_predictions = segments_predictions.reshape(-1, 2)
                    gt = gt.reshape(-1, 2)
                    giou_loss = giou_1d_loss(segments_predictions, gt)
                    loss = outputs.loss + (l1_loss + giou_loss)
                elif model_args.loss_type == "l2_giou_loss":
                    l2_loss = nn.MSELoss()(segments_predictions, gt)
                    normalized_l2_loss = l2_loss / num_segments
                    l2_loss = normalized_l2_loss * self.model.model_args.loss_weight
                    segments_predictions = segments_predictions.reshape(-1, 2)
                    gt = gt.reshape(-1, 2)
                    giou_loss = giou_1d_loss(segments_predictions, gt)
                    loss = outputs.loss + (l2_loss + giou_loss) 


                output = (outputs.logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output




    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("VTimeLLM", VTimeLLMConfig)
AutoModelForCausalLM.register(VTimeLLMConfig, VTimeLLMLlamaForCausalLM)
