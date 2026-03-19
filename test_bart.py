import torch
from transformers import BartForConditionalGeneration, AutoTokenizer

t = AutoTokenizer.from_pretrained('jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b')
print("Model eos:", t.eos_token_id, "bos:", t.bos_token_id, "pad:", t.pad_token_id)
print("Config decoder_start_token_id:", BartForConditionalGeneration.from_pretrained('jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b').config.decoder_start_token_id)

labels = torch.tensor([[135, 104, 4, 89]])
model = BartForConditionalGeneration.from_pretrained('jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b')
out = model(input_ids=torch.tensor([[10, 20, 30]]), labels=labels)

print("HF shift right output:")
print(out.logits.shape)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

print("True decoder inputs built by HF:", shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id))

