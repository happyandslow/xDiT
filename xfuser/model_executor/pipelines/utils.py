# from https://github.com/chengzeyi/ParaAttention/blob/main/examples/run_hunyuan_video.py
import functools
from typing import Any, Dict, Union, Optional
import logging
import time

import torch

from diffusers import DiffusionPipeline, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND


from xfuser.core.distributed import (
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)

from xfuser.model_executor.layers.attention_processor import xFuserHunyuanVideoAttnProcessor2_0

assert xFuserHunyuanVideoAttnProcessor2_0 is not None


def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logging.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        assert batch_size % get_classifier_free_guidance_world_size(
        ) == 0, f"Cannot split dim 0 of hidden_states ({batch_size}) into {get_classifier_free_guidance_world_size()} parts."

        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states,
                                                      timestep,
                                                      encoder_attention_mask)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1)
        hidden_states = hidden_states.flatten(1, 3)

        hidden_states = torch.chunk(hidden_states,
                                    get_classifier_free_guidance_world_size(),
                                    dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states,
                                    get_sequence_parallel_world_size(),
                                    dim=-2)[get_sequence_parallel_rank()]

        encoder_attention_mask = encoder_attention_mask[0].to(torch.bool)
        encoder_hidden_states_indices = torch.arange(
            encoder_hidden_states.shape[1],
            device=encoder_hidden_states.device)
        encoder_hidden_states_indices = encoder_hidden_states_indices[
            encoder_attention_mask]
        encoder_hidden_states = encoder_hidden_states[
            ..., encoder_hidden_states_indices, :]
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size(
        ) != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        encoder_hidden_states = torch.chunk(
            encoder_hidden_states,
            get_classifier_free_guidance_world_size(),
            dim=0)[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states,
                get_sequence_parallel_world_size(),
                dim=-2)[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = image_rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=0)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):

                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None,
                    image_rotary_emb)

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None,
                    image_rotary_emb)

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)
        hidden_states = get_cfg_group().all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.reshape(batch_size,
                                              post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, -1, p_t, p, p)

        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states, )

        return Transformer2DModelOutput(sample=hidden_states)

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
        block.attn.processor = xFuserHunyuanVideoAttnProcessor2_0()

