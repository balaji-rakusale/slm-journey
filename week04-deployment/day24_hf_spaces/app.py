import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading model...")
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
model.eval()
print("Model loaded")

def generate_response(instruction, input_text, max_tokens, temperature):
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(prompt):].strip()
    return response

# Build Gradio UI
with gr.Blocks(title="SLM Demo") as demo:
    gr.Markdown("""
    #Small Language Model Demo
    ### Built from scratch in 24 days at 1.5 hrs/day
    """)

    with gr.Row():
        with gr.Column():
            instruction = gr.Textbox(
                label="Instruction",
                placeholder="What do you want the model to do?",
                value="Classify this news article into a category."
            )
            input_text = gr.Textbox(
                label="Input (optional)",
                placeholder="Provide context or text here...",
                value="Apple launches new AI powered MacBook today."
            )
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    label="Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    label="Temperature"
                )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Textbox(
                label="Model Response",
                lines=10
            )

    generate_btn.click(
        fn=generate_response,
        inputs=[instruction, input_text, max_tokens, temperature],
        outputs=output
    )

    gr.Markdown("""
    ### Example Instructions:
    - Classify this news article into a category.
    - What is artificial intelligence?
    - Summarize the following text:
    - What is machine learning?
    """)

if __name__ == "__main__":
    demo.launch()