import gradio as gr

import os
import gradio as gr
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from transformers import pipeline
cache_dir="E:\Cache\Hugging_Face"

project_path = os.path.abspath(os.path.relpath('../../../', os.getcwd()))
data_path = os.path.join(project_path, 'FT4LLM/Data')
prompt_path = os.path.join(data_path, 'prompt')



def get_prompt(language):
    if language=="en":
        return "You are a practitioner in the epilepsy treatment industry. Try best to complete the user's instruction given to you. Be concise and professional."
    else:
        return "你是癫痫康复行业的专业人士，请尽力完成用户的指令，并保持输出专业和简短"


def main():
    def conversation_fn(message, history):
        messages_map = [
            {"role": "system", "content": get_prompt("en")},
        ]
        for history_item in history:
            messages_map.append({"role": "user", "content": history_item[0]})
            messages_map.append({"role": "assistant", "content": history_item[1]})
        messages_map.append({"role": "user", "content": message})
        input_processed = Coversation_epipaca.tokenizer.apply_chat_template(
            messages_map,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = Coversation_epipaca(
            input_processed,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][len(input_processed):]
    # load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained("hfl/llama-3-chinese-8b-instruct", quantization_config=bnb_config,
                                                      device_map='auto',
                                                      cache_dir=cache_dir)
    model = PeftModel.from_pretrained(base_model, "CocoNutZENG/Epipaca")

    tokenizer = AutoTokenizer.from_pretrained("CocoNutZENG/Epipaca", padding_side="right", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token


    Coversation_epipaca = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    terminators = [
        Coversation_epipaca.tokenizer.eos_token_id,
        Coversation_epipaca.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    # Conversation pipeline for inference
    gr.ChatInterface(
        conversation_fn,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
        title="Yes Man",
        description="Ask Yes Man any question",
        theme="soft",
        clear_btn="Clear",
    ).launch()





if __name__ == '__main__':
    main()