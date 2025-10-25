import csv
import shutil
import os
from google import genai
from PyPDF2 import PdfReader
from google.genai import types
import time

# Konfiguration
GEMINI_API_KEY = "pimmel"
INPUT_FOLDER = '../new_pdf'
OUTPUT_FOLDER = '../generated_texts'
OUTPUT_CSV = './GeminiOutput.csv'
OUTPUT_UNFORMATTED = './GeminiOutput_unformatted.csv'
PROCESSED_FOLDER = '../used_texts'
USED_PDFS = '../used_pdf'
SYSTEM_PROMPT_PATH = '../system_prompt.txt'
VALIDATION_PROMPT = "../validation_prompt.txt"
#MAIN_PROMPT = 'Hier ist ein Wissenschaftlicher Artikel, bitte halte dich genau an das Ausgabeformat: <paper>{}</paper>'
MAIN_PROMPT = 'Here is a scientific article, please stick exactly to the output format: <paper>{}</paper>'
# Initialisierung des AI-Clients
client = genai.Client(api_key=GEMINI_API_KEY)

def ensure_directories_exist():
    """Stellt sicher, dass alle benötigten Verzeichnisse existieren."""
    for directory in [INPUT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER, USED_PDFS]:
        os.makedirs(directory, exist_ok=True)

def convert_pdf_to_text(pdf_path, output_path):
    """Konvertiert eine einzelne PDF-Datei in Text."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ''.join(page.extract_text() for page in pdf_reader.pages)

    with open(output_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

def convert_all_pdfs():
    """Konvertiert alle PDF-Dateien im Eingabeordner sequentiell."""
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    for filename in pdf_files:
        pdf_path = os.path.join(INPUT_FOLDER, filename)
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        if os.path.exists(output_path):
            print(f'Die Datei {output_filename} existiert bereits. Überspringe Konvertierung.')
            continue

        convert_pdf_to_text(pdf_path, output_path)
        print(f'Die PDF-Datei {filename} wurde erfolgreich konvertiert.')

        processed_path = os.path.join(USED_PDFS, filename)
        shutil.move(pdf_path, processed_path)
        print(f"PDF-Datei {filename} wurde verschoben.")

def read_system_prompt():
    """Liest den Systempromt aus der Datei system_prompt.txt."""
    try:
        with open(SYSTEM_PROMPT_PATH, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {SYSTEM_PROMPT_PATH}: {e}")
        return None

def read_validation_prompt():
    """Liest den Validierungsprompt aus der Datei validation_prompt.txt."""
    try:
        with open(VALIDATION_PROMPT, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {VALIDATION_PROMPT}: {e}")
        return None

def send_text_to_gemini(text, prompt):
    """Sendet Text an die Gemini API und gibt die Antwort zurück."""
    system_prompt = read_system_prompt()
    if not system_prompt:
        print("Fehler: System-Prompt konnte nicht gelesen werden.")
        return None
    try:
        full_prompt = f"{system_prompt}\n\n{prompt.format(text)}"
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=5120, include_thoughts=True)
            ),
        )
        print(response)
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"Fehler bei der API-Anfrage: {e}")
        return None

def validate_gemini_response(initial_response, original_text):
    """Validiert die erste Antwort von Gemini mit einer zweiten KI-Anfrage"""
    validation_prompt = read_validation_prompt()
    if not validation_prompt:
        print("Fehler: Validierungs-Prompt konnte nicht gelesen werden.")
        return None
    try:
        full_prompt = f"""{validation_prompt}

        Here is a scientific article: {original_text}

        Here is a preliminary evaluation that you should critically evaluate again 
        and output in the specified format: {initial_response}"""

        print("Sende Validierungs-Anfrage an Gemini API...")

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=5120, include_thoughts=False)
            ),
        )

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"Fehler bei der API-Anfrage: {e}")
        return None

def extract_formatted_output(text):
    """Extrahiert den formatierten Output aus dem KI-Antworttext."""
    cleaned_text = text.strip()
    cleaned_text = cleaned_text.strip('"')
    lines = cleaned_text.split('\n')

    for line in lines:
        line = line.strip()
        parts = line.split(',')
        if len(parts) == 12:
            if all(part.replace('.', '').isdigit() for part in parts[4:]):
                return line

    print("Konnte keinen gültigen formatierten Output finden.")
    return None

def save_unformatted_output(filename, content):
    """Speichert unformatierten Output in eine separate CSV-Datei."""
    with open(OUTPUT_UNFORMATTED, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([filename, content])
    print(f'Unformatierter Output für {filename} wurde in {OUTPUT_UNFORMATTED} gespeichert.')

def process_text_file(filename):
    """Verarbeitet eine einzelne Textdatei."""
    txt_path = os.path.join(OUTPUT_FOLDER, filename)
    with open(txt_path, 'r', encoding='utf-8') as txt_file:
        text = txt_file.read()

    initial_result = send_text_to_gemini(text, MAIN_PROMPT)
    if initial_result:
        time.sleep(30)
        validated_result = validate_gemini_response(initial_result, text)
        final_result = validated_result if validated_result else initial_result

        formatted_output = extract_formatted_output(final_result)
        if formatted_output:
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([formatted_output])
            print(f'Die Datei {filename} wurde erfolgreich verarbeitet und der formatierte Output extrahiert.')
        else:
            save_unformatted_output(filename, final_result)
            print(f'Konnte keinen gültigen formatierten Output für {filename} extrahieren. Unformatierter Output gespeichert.')

        processed_path = os.path.join(PROCESSED_FOLDER, filename)
        shutil.move(txt_path, processed_path)
        print(f"Datei {filename} wurde verarbeitet und verschoben.")
    else:
        print(f'Fehler bei der Verarbeitung der Datei {filename}. Kein Output verfügbar.')

def process_all_texts():
    """Verarbeitet alle Textdateien im Ausgabeordner sequentiell."""
    txt_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.lower().endswith('.txt')]
    for filename in txt_files:
        process_text_file(filename)
        time.sleep(30)


def main():
    """Hauptfunktion zur Ausführung der Pipeline."""
    ensure_directories_exist()
    convert_all_pdfs()
    process_all_texts()

if __name__ == "__main__":
    main()