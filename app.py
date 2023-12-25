import gradio as gr
import fitz
from PIL import Image
import io
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_vision = genai.GenerativeModel('gemini-pro-vision')
model_text = genai.GenerativeModel("gemini-pro")


# Global variable or file path to store intermediate data
INTERMEDIATE_JSON_PATH = "intermediate_data.json"
INTERMEDIATE_JOB_DESC_PATH = "intermediate_job_desc.txt"


def load_prompt(filename):
    """Function to load a prompt from a file"""
    try:
        with open(filename, "r") as file:
            return file.read()
    except Exception as e:
        return f"Error loading prompt: {e}"


def process_pdf_and_save_job_desc(pdf_file, job_description):
    if not pdf_file:
        return None, "No file provided"

    # Convert PDF to image
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    doc.close()

    # Further processing with the image
    # Substitute this with the actual computation or model invocation
    # For example, using the genai model:
    # Define the prompt you want to use
    prompt = load_prompt("prompts/resume_parsing_prompt.txt")
    response = model_vision.generate_content([prompt, image])
    json_data = response.text

    # Save the JSON data for other tabs to access
    with open(INTERMEDIATE_JSON_PATH, "w") as json_file:
        json.dump(json_data, json_file)

    with open(INTERMEDIATE_JOB_DESC_PATH, "w") as file:
        file.write(job_description)

    return image, json_data


def generate_interview_questions():
    # Assuming json_data is a string containing JSON data
    # Here, you would customize the prompt to include specific details
    # from the json_data or pass it directly as part of the prompt.
    with open(INTERMEDIATE_JSON_PATH, "r") as json_file:
        json_data = json.load(json_file)
    prompt = load_prompt("prompts/interview_questions_prompt.txt") + json_data

    # Generate responses using the model
    responses = model_text.generate_content(prompt)

    # Return the generated questions or content
    return responses.text


# Define the new Gradio interface for generating interview questions
interview_interface = gr.Interface(
    fn=generate_interview_questions,
    inputs=[],
    outputs=gr.Textbox(label="Generated Interview Questions"),
    title="Generate Interview Questions"
)


def generate_skill_gap_analysis():
    try:
        # Read the saved resume data (JSON)
        with open(INTERMEDIATE_JSON_PATH, "r") as file:
            json_data = file.read()

        # Read the saved job description
        with open(INTERMEDIATE_JOB_DESC_PATH, "r") as file:
            job_description = file.read()

        # Construct a detailed prompt for the Gemini model
        prompt = load_prompt("prompts/skills_gap_prompt.txt").replace(
            "job_description", job_description).replace("json_data", json_data)
        # Call the Gemini model to generate the skill gap analysis
        response = model_text.generate_content(prompt)

        # Format and return the skill gap analysis
        return response.text

    except Exception as e:
        return f"An error occurred: {e}"


skill_gap_analysis_interface = gr.Interface(
    fn=generate_skill_gap_analysis,
    inputs=[],  # No inputs
    outputs=gr.Textbox(label="Skill Gap Analysis"),
    title="Skill Gap Analysis"
)


def generate_cover_letter():
    try:
        # Read the saved job description
        with open(INTERMEDIATE_JOB_DESC_PATH, "r") as file:
            job_description = file.read()

        # Read the saved resume data (JSON)
        with open(INTERMEDIATE_JSON_PATH, "r") as file:
            json_data = file.read()

        # Create a prompt for the cover letter
        prompt = load_prompt("prompts/cover_letter_prompt.txt").replace(
            "job_description", job_description).replace("json_data", json_data)

        # Generate the cover letter using the model
        response = model_text.generate_content(prompt, stream=True)
        response.resolve()

        return response.text

    except Exception as e:
        return f"An error occurred: {e}"


# Define the Gradio interface for generating a cover letter
cover_letter_interface = gr.Interface(
    fn=generate_cover_letter,
    inputs=[],
    outputs=gr.Textbox(label="Generated Cover Letter"),
    title="Cover Letter Generator"
)


def display_json():
    # Read and return the JSON data saved by the first tab
    try:
        with open(INTERMEDIATE_JSON_PATH, "r") as json_file:
            json_data = json.load(json_file)
        return json_data
    except FileNotFoundError:
        return "No data available yet. Please run the first tab."


def gradio_pdf_interface(pdf_content, job_description):
    # Wrapper patchwork :c
    image, _ = process_pdf_and_save_job_desc(
        pdf_content, job_description)  # Get both but only use image
    return image


# Define individual interfaces for each tab
pdf_interface = gr.Interface(
    fn=gradio_pdf_interface,
    inputs=[gr.File(type="binary"), gr.Textbox(label="Job Description")],
    outputs=gr.Image(),
    title="PDF Processing and Job Description"
)
json_interface = gr.Interface(
    fn=display_json, inputs=None, outputs=gr.Textbox(), title="Display JSON")

# Combine interfaces into a TabbedInterface
demo = gr.TabbedInterface([pdf_interface, json_interface, interview_interface, cover_letter_interface, skill_gap_analysis_interface], [
                          "Process PDF", "JSON Output", "Interview Questions", "Cover Letter", "Skill Gap Analysis"])

if __name__ == "__main__":
    demo.launch()
