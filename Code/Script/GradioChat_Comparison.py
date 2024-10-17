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

Coversation_epipaca=None
Coversation_original=None
terminators=None

def get_prompt(language):
    if language=="en":
        return "You are a practitioner in the epilepsy treatment industry. Try best to complete the user's instruction given to you. Be concise and professional."
    else:
        return "你是癫痫康复行业的专业人士，请尽力完成用户的指令，并保持输出专业和简短"


def main():
    def compare_models(message):
        if ('\u0041' <= message[0] <= '\u005a') or ('\u0061' <= message[0] <= '\u007a'):
            language = 'en'
        else:
            language = 'zh'
        messages_map = [
            {"role": "system", "content": get_prompt(language=language)},
        ]
        messages_map.append({"role": "user", "content": message})
        input_processed = Coversation_epipaca.tokenizer.apply_chat_template(
            messages_map,
            tokenize=False,
            add_generation_prompt=True,

        )
        output1 = Coversation_original(input_processed, max_new_tokens=256,
                                       eos_token_id=terminators,
                                       do_sample=True,
                                       temperature=0.2,
                                       top_p=0.8)[0]['generated_text'][len(input_processed):]
        output2 = Coversation_epipaca(input_processed, max_new_tokens=256,
                                      eos_token_id=terminators,
                                      do_sample=True,
                                      temperature=0.2,
                                      top_p=0.8)[0]['generated_text'][len(input_processed):]
        return output1, output2
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
    # Conversation pipeline for inference

    Coversation_epipaca = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    Coversation_original = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    terminators = [
        Coversation_epipaca.tokenizer.eos_token_id,
        Coversation_original.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    # 创建Gradio界面
    with gr.Blocks() as demo:
        gr.Markdown("## Epilepsy LLM Comparison")
        with gr.Row():
            with gr.Column():
                output1 = gr.Textbox(label="LLAMA-3 Chinese", lines=10)
            with gr.Column():
                output2 = gr.Textbox(label="Epipaca", lines=10)
        with gr.Row():
            user_input = gr.Textbox(label="Input your word here")
        compare_button = gr.Button("Send")
        compare_button.click(compare_models, inputs=user_input, outputs=[output1, output2])
    demo.launch()



if __name__ == '__main__':
    main()