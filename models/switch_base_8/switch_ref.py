import torch
from pathlib import Path
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

folder = Path(__file__).with_name("switch-base-8")
tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    folder,
    local_files_only=True,
    dtype=torch.float32,
).eval()

text = (
    "She packed her suitcase, went to the airport, and <extra_id_0>. "
    "A few hours later, she arrived in Paris."
)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=33,
        do_sample=False,
        num_beams=1,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
