from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import torch
import gradio as gr

#loading slm

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading TinyLlama model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print("TinyLlama Loaded Successfully!")

#loading embeding model

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")

#global var

chunks = []
index = None

#pdf orocessing func

def process_pdf(pdf_file):

    global chunks
    global index

    try:

        reader = PdfReader(pdf_file.name)

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

        if len(text.strip()) == 0:
            return "No readable text found in PDF."

        # Split into chunks
        chunk_size = 500

        chunks = []

        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

        # Create embeddings
        embeddings = embedding_model.encode(chunks)

        # Create FAISS index
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(np.array(embeddings))

        return f"PDF processed successfully! Total chunks created: {len(chunks)}"

    except Exception as e:
        return f"Error processing PDF: {str(e)}"

#ques answer fn

def ask_question(query):

    global chunks
    global index

    try:

        if index is None:
            return "Please upload and process a PDF first."

        # Create query embedding
        query_embedding = embedding_model.encode([query])

        # Search relevant chunks
        D, I = index.search(np.array(query_embedding), k=3)

        relevant_text = ""

        for i in I[0]:
            relevant_text += chunks[i] + "\n"

        # Prompt for AI
        final_prompt = f"""
You are an AI study assistant.

Use the context below to answer the question in simple and easy words.

Context:
{relevant_text}

Question:
{query}

Answer:
"""

        # Generate response
        response = generator(
            final_prompt,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

        answer = response[0]["generated_text"]

        # Remove prompt from output
        answer = answer.replace(final_prompt, "")

        return answer.strip()

    except Exception as e:
        return f"Error generating answer: {str(e)}"

#gradio ui

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # AI Exam Preparation Assistant
        
        Upload your study notes.
        """
    )

    with gr.Row():

        pdf_input = gr.File(label="Upload PDF")

        process_button = gr.Button("Process PDF")

    process_output = gr.Textbox(label="PDF Status")

    process_button.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=process_output
    )

    gr.Markdown("Ask Questions")

    question_input = gr.Textbox(
        label="Enter your question",
        placeholder="Example: Explain Artificial Intelligence in simple words"
    )

    ask_button = gr.Button("Ask AI")

    answer_output = gr.Textbox(
        label="AI Answer",
        lines=10
    )

    ask_button.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )

demo.launch()