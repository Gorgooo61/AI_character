import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import LLM_models, LLM_params


class LlamaWrapper:
    def __init__(self):
        self.model_name = LLM_models["meta_model"]
        self.max_tokens = LLM_params["max_tokens"]
        self.temperature = LLM_params["temperature"]
        self.top_p = LLM_params["top_p"]

        # compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        compute_dtype = torch.bfloat16 # cuda test
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=compute_dtype
        )
        self.model.eval()


    # HF AutoTokenizer chat template builder, this might be temporary
    def _build_chat_prompt(self, system_prompt, user_prompt):
        system_prompt = (system_prompt or "").strip()
        user_prompt = (user_prompt or "").strip()

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        if system_prompt:
            return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        return f"User: {user_prompt}\nAssistant:"


    def generate(self, system_prompt, user_prompt, max_new_tokens=None, temperature=None, top_p=None):
            prompt = self._build_chat_prompt(system_prompt, user_prompt)

            max_new_tokens = self.max_tokens if max_new_tokens is None else int(max_new_tokens)
            temperature = self.temperature if temperature is None else float(temperature)
            top_p = self.top_p if top_p is None else float(top_p)

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True, # set temper, top_p
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = output[0][input_len:] # only new tokens -> cut out the prompt
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text.strip()