import gradio as gr
from main import app, initialize_state  # assuming these are defined in main.py
import json
import tempfile

def process_question(user_question):
    # Step 1: Invoke the graph with initial state
    state = initialize_state(user_question)
    final_state = app.invoke(state)

    # Step 2: Extract data to show in UI
    original = final_state.get("original_question", "")
    answers = final_state.get("answers", [])

    # Format answers row-wise
    answer_blocks = ""
    for idx, qa in enumerate(answers, 1):
        answer_blocks += f"**{idx}. Q:** {qa['question']}\n\n**A:**\n```\n{qa['answer']}\n```\n\n"

    # Step 3: Save full final_state dict as a JSON file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as temp_file:
        json.dump(final_state, temp_file, indent=2)
        temp_path = temp_file.name

    return original, answer_blocks, temp_path


# Build the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## **Recursive Data Assistant**\nAsk questions about the store sales data. The system will reason and resolve complex queries.\n")

    user_input = gr.Textbox(label="Enter your question", placeholder="e.g., What are the top-selling products by revenue in each store?")
    submit_btn = gr.Button("Run Query")

    output_question = gr.Textbox(label="Original Question", interactive=False)
    output_answers = gr.Markdown(label="Sub-questions and Answers")

    json_file_output = gr.File(label="Download Full Execution State (.json)")

    submit_btn.click(
        fn=process_question,
        inputs=user_input,
        outputs=[output_question, output_answers, json_file_output]
    )

demo.launch()
