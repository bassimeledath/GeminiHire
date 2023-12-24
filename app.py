import gradio as gr
import fitz
from PIL import Image
import io
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))
model_vision = genai.GenerativeModel('gemini-pro-vision')
model_text = genai.GenerativeModel("gemini-pro")


# Global variable or file path to store intermediate data
INTERMEDIATE_JSON_PATH = "intermediate_data.json"
INTERMEDIATE_JOB_DESC_PATH = "intermediate_job_desc.txt"


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
    prompt = """For the attached resume, give the information in the following json format:
{
  \"Education\": [
    {
      \"Institution\": \"<Name of Institution>\",
      \"Degree\": \"<Degree Obtained>\",
      \"Field of Study\": \"<Field of Study>\",
      \"Start Date\": \"<Start Date>\",
      \"End Date\": \"<End Date or \'Present\'>\"
    }
    // Additional entries if more than one degree
  ],
  \"Work Experience\": [
    {
      \"Company\": \"<Company Name>\",
      \"Role\": \"<Job Title>\",
      \"Start Date\": \"<Start Date>\",
      \"End Date\": \"<End Date or \'Present\'>\",
      \"Responsibilities\": \"<Key Responsibilities and Achievements>\"
    }
    // Additional entries for more work experiences
  ],
  \"Skills\": \"<List of skills>\"
}
Note: The model should be able to handle cases where some sections (like Projects or Certifications) are not present in the resume. In such cases, these sections should be omitted from the JSON output."""
    response = model_vision.generate_content([prompt, image])
    json_data = response.text

    # Save the JSON data for other tabs to access
    with open(INTERMEDIATE_JSON_PATH, "w") as json_file:
        json.dump(json_data, json_file)

    with open(INTERMEDIATE_JOB_DESC_PATH, "w") as file:
        file.write(job_description)

    return image


def generate_interview_questions():
    # Assuming json_data is a string containing JSON data
    # Here, you would customize the prompt to include specific details
    # from the json_data or pass it directly as part of the prompt.
    with open(INTERMEDIATE_JSON_PATH, "r") as json_file:
        json_data = json.load(json_file)
    prompt = "Give three interview questions based on the following work experience data: " + json_data

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
        prompt = f"Analyze the skill gaps for the following candidate based on their resume data and the job description provided. \n\nResume Data: {json_data}\n\nJob Description: {job_description}\n\nProvide a detailed skill gap analysis and recommendations for improvement."

        # Call the Gemini model to generate the skill gap analysis
        response = model_text.generate_content(prompt, stream=True)
        response.resolve()

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
        prompt = f"Create a cover letter based on this job description: {job_description} and the candidate's details: {json_data}"

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


# Define individual interfaces for each tab
pdf_interface = gr.Interface(
    fn=process_pdf_and_save_job_desc,
    inputs=[gr.File(type="binary"), gr.Textbox(label="Job Description")],
    outputs=gr.Image(),  # Adjust as per your existing outputs
    title="PDF Processing and Job Description"
)
json_interface = gr.Interface(
    fn=display_json, inputs=None, outputs=gr.Textbox(), title="Display JSON")

# Combine interfaces into a TabbedInterface
demo = gr.TabbedInterface([pdf_interface, json_interface, interview_interface, cover_letter_interface, skill_gap_analysis_interface], [
                          "Process PDF", "JSON Output", "Interview Questions", "Cover Letter", "Skill Gap Analysis"])

if __name__ == "__main__":
    demo.launch()
