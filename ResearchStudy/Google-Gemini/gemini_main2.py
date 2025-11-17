import csv
import shutil
import os
from google import genai
from PyPDF2 import PdfReader
from google.genai import types
import time

# --- Configuration ---
GEMINI_API_KEY = "birne"  # IMPORTANT: Replace with your actual Gemini API key
INPUT_FOLDER = '../new_pdf'
OUTPUT_FOLDER = '../generated_texts'
PROCESSED_FOLDER = '../used_texts'
USED_PDFS = '../used_pdf'

# --- Prompt File Paths ---
QUALITY_PROMPT_PATH = '../quality_prompt.txt'
CONTENT_PROMPT_PATH = "../content_prompt.txt"  # Corrected variable name from original

# --- Output CSV File Paths for the new two-step process ---
QUALITY_OUTPUT_CSV = './Quality_Output.csv'
QUALITY_OUTPUT_UNFORMATTED_CSV = './Quality_Output_unformatted.csv'
CONTENT_OUTPUT_CSV = './Content_Output.csv'
CONTENT_OUTPUT_UNFORMATTED_CSV = './Content_Output_unformatted.csv'

# Main prompt template to wrap the paper's text
MAIN_PROMPT = 'Here is a scientific article, please stick exactly to the output format: <paper>{}</paper>'

# --- AI Client Initialization ---
client = genai.Client(api_key=GEMINI_API_KEY)


def ensure_directories_exist():
    """Ensures that all necessary directories exist."""
    for directory in [INPUT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER, USED_PDFS]:
        os.makedirs(directory, exist_ok=True)


def convert_pdf_to_text(pdf_path, output_path):
    """Converts a single PDF file to text."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ''.join(page.extract_text() for page in pdf_reader.pages)

    with open(output_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)


def convert_all_pdfs():
    """Converts all PDF files in the input folder sequentially."""
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    for filename in pdf_files:
        pdf_path = os.path.join(INPUT_FOLDER, filename)
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        if os.path.exists(output_path):
            print(f'File {output_filename} already exists. Skipping conversion.')
            continue

        convert_pdf_to_text(pdf_path, output_path)
        print(f'The PDF file {filename} was successfully converted.')

        # Note: Moving the PDF is now handled after all processing is done for that file.


def read_prompt(prompt_path):
    """Reads a prompt from the specified file path."""
    try:
        with open(prompt_path, "r", encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {prompt_path}: {e}")
        return None


def send_request_to_gemini(text, instructions_prompt):
    """
    Sends text to the Gemini API with specific instructions and returns the response.
    This function combines the instructions and the main text wrapper for the API call.
    """
    if not instructions_prompt:
        print("Error: Instructions prompt could not be read.")
        return None
    try:
        # This structure matches the original script's logic
        full_prompt = f"{instructions_prompt}\n\n{MAIN_PROMPT.format(text)}"

        # The API call is kept exactly as it was in the original script
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=5120, include_thoughts=True)
            ),
        )
        print(response)  # For debugging, as in the original script
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"An error occurred during the API request: {e}")
        return None


def extract_formatted_output(text):
    """Extracts the formatted output from the AI response text."""
    if not text:
        return None
    cleaned_text = text.strip().strip('"')
    lines = cleaned_text.split('\n')

    for line in lines:
        line = line.strip()
        parts = line.split(',')
        if len(parts) == 12:
            # Simple validation from original script
            if all(part.strip().replace('.', '', 1).isdigit() for part in parts[4:]):
                return line

    print("Could not find a valid formatted output.")
    return None


def save_output(filename, content, formatted_csv_path, unformatted_csv_path):
    """Saves formatted or unformatted output to the respective CSV files."""
    formatted_output = extract_formatted_output(content)

    if formatted_output:
        with open(formatted_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # The original code saved the entire line as a single CSV field
            writer.writerow([formatted_output])
        print(f'Formatted output for {filename} has been saved to {formatted_csv_path}.')
    else:
        with open(unformatted_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([filename, content])
        print(f'Unformatted output for {filename} has been saved to {unformatted_csv_path}.')


def process_all_texts():
    """Processes all text files in the output folder sequentially."""
    quality_prompt = read_prompt(QUALITY_PROMPT_PATH)
    content_prompt = read_prompt(CONTENT_PROMPT_PATH)

    if not quality_prompt or not content_prompt:
        print("Aborting: Could not read quality or content prompt files.")
        return

    txt_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.lower().endswith('.txt')]
    for filename in txt_files:
        print(f"\n--- Processing file: {filename} ---")
        txt_path = os.path.join(OUTPUT_FOLDER, filename)
        with open(txt_path, 'r', encoding='utf-8') as txt_file:
            text_content = txt_file.read()

        # --- Step 1: Quality Assessment ---
        print("Requesting Quality Assessment...")
        quality_result = send_request_to_gemini(text_content, quality_prompt)
        if quality_result:
            save_output(filename, quality_result, QUALITY_OUTPUT_CSV, QUALITY_OUTPUT_UNFORMATTED_CSV)
        else:
            print(f'Failed to get quality assessment for {filename}.')

        time.sleep(30)  # Delay between API calls as in original script

        # --- Step 2: Content Extraction ---
        print("Requesting Content Extraction...")
        # This is a separate request, using only the original paper text as input
        content_result = send_request_to_gemini(text_content, content_prompt)
        if content_result:
            save_output(filename, content_result, CONTENT_OUTPUT_CSV, CONTENT_OUTPUT_UNFORMATTED_CSV)
        else:
            print(f'Failed to get content extraction for {filename}.')

        # --- Step 3: Move Processed Files ---
        print(f"Moving processed files for {filename}...")
        # Move the processed .txt file
        processed_text_path = os.path.join(PROCESSED_FOLDER, filename)
        shutil.move(txt_path, processed_text_path)

        # Move the original .pdf file
        pdf_filename = os.path.splitext(filename)[0] + '.pdf'
        original_pdf_path = os.path.join(INPUT_FOLDER, pdf_filename)
        processed_pdf_path = os.path.join(USED_PDFS, pdf_filename)
        if os.path.exists(original_pdf_path):
            shutil.move(original_pdf_path, processed_pdf_path)

        print(f"File {filename} and its PDF have been processed and moved.")

        time.sleep(30)  # Delay between processing different files


def main():
    """Main function to run the pipeline."""
    ensure_directories_exist()
    convert_all_pdfs()
    process_all_texts()


if __name__ == "__main__":
    main()
