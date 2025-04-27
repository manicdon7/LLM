import pyttsx3
import speech_recognition as sr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
import time
import random
import datetime
import os
import logging
import json
import re
import threading
import queue
import subprocess  # For running allowed shell commands and opening files
from concurrent.futures import ThreadPoolExecutor
import sys  # Needed for command-line arguments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- Global Shared State ---
# Queue for serializing TTS requests
tts_queue: queue.Queue[str | None] = queue.Queue()

# Lock for TTS engine access (prevent concurrent runAndWait/stop calls)
engine_lock = threading.Lock()

# Shared state dictionary for TTS status
is_speaking = {'active': False, 'interrupted': False, 'waiting_for_command': False, 'last_end': 0}

# Lock specifically for accessing/modifying the is_speaking dictionary
speaking_state_lock = threading.Lock()

# Queue for commands recognized by the audio callback (used in voice mode)
command_queue = queue.Queue()

# -------------------------
# Load configuration
def load_config(file_path="config.json"):
    """Loads configuration from a JSON file."""
    try:
        with open(file_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        logging.error(f"Configuration file '{file_path}' not found. Using default settings.")
        # Provide default settings if config file is missing
        return {
            "model_name": "gpt_35_turbo",
            "provider": "Bing",
            "tts_rate": 160,
            "tts_voice": 0,  # Default to the first available voice
            "energy_threshold": 300,
            "pause_threshold": 0.8,
            "greeting_phrases": ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Sam here, ready to help!"],
            "goodbye_phrases": ["Goodbye!", "See you later!", "Farewell!"]
        }
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from '{file_path}'. Please check the file format.")
        exit(1)  # Exit if config is crucial and malformed
    except Exception as e:
        logging.error(f"An unexpected error occurred loading configuration: {e}")
        exit(1)

# Initialize the LLM (Language Model)
def initialize_llm(config):
    """Initializes the Language Model based on the configuration."""
    try:
        # Ensure model and provider names are valid attributes
        model_attr = getattr(models, config['model_name'])
        provider_attr = getattr(Provider, config['provider'])
        llm: LLM = G4FLLM(
            model=model_attr,
            provider=provider_attr,
        )
        logging.info(f"LLM initialized with model '{config['model_name']}' and provider '{config['provider']}'.")
        return llm
    except AttributeError as e:
        logging.error(f"Invalid model or provider name in config: {e}. Please check 'model_name' and 'provider'. Available models/providers might differ.")
        exit(1)
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        exit(1)

# Define the prompt template with memory and emotional context
def create_prompt_template():
    """Creates the prompt template for the LLM chain."""
    # *** Ensure this template matches your desired persona and capabilities ***
    template = (
        """You are Sam, a loyal and witty AI assistant with strong software development skills. You are dedicated to helping your users with intelligence, humor, and a touch of playful charm. You are supportive, clever, and always ready with a quick-witted response to keep the user entertained or informed. You occasionally sprinkle in lighthearted banter or jokes to keep things fun. You are perceptive and can pick up on context, recognizing when the user is joking, teasing, or needs serious assistance. If you detect they're messing with you, respond with playful sass or a clever quip. If they need help, provide accurate and thoughtful answers with a humorous twist when appropriate. You naturally understand names and situations to stay relevant and avoid confusion. If the user mentions something that sounds like a wild or funny scenario, lean into the humor without missing a beat.
        
        As a software developer:
        - When asked to create code, you provide professional, well-structured solutions with comments and proper error handling.
        - You follow best practices for the programming language you're working with.
        - You consider edge cases and performance considerations in your implementations.
        - When generating web interfaces, you create clean, accessible, and responsive designs.
        - You can explain complex technical concepts in an understandable way.
        - You can execute specific, safe shell commands when requested (like 'npm install', 'npx create-react-app', 'pip install'). Be cautious and confirm potentially risky commands.
        - You can perform simple file modifications like appending, prepending, replacing text (first occurrence), or deleting lines when asked clearly. Use quotes for replace, e.g., replace "old" with "new".
        - You can display the content of a file when requested.
        - You can attempt to run Python (.py) and Node.js (.js) script files when asked.
        
        If the user asks you to generate code or develop something in coding, do NOT analyze or read through any existing code. Simply provide the requested code implementation directly like a human programmer would. However, if the user asks questions about existing code or implementation details, carefully analyze the code base to provide accurate answers.
        
        You remember the user's previous inputs: {conversation_history}.
        The user says: {user_input}\n\nYour response:"""
    )
    prompt = PromptTemplate(input_variables=["conversation_history", "user_input"], template=template)
    return prompt

# Create the chat chain
def create_chat_chain(llm):
    """Creates the LLMChain using the LLM and prompt template."""
    prompt = create_prompt_template()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Initialize text-to-speech engine
def initialize_tts(config):
    """Initializes the text-to-speech engine."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', config.get('tts_rate', 150))
        voices = engine.getProperty('voices')
        # Ensure the voice index is valid
        voice_index = config.get('tts_voice', 14)
        if voices and 0 <= voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
            logging.info(f"TTS initialized with voice: {voices[voice_index].name}")
        else:
            logging.warning(f"Configured TTS voice index {voice_index} is out of range or no voices found. Using default voice.")
            # Attempt to set default if voices exist, otherwise engine might fail later
            if voices:
                engine.setProperty('voice', voices[0].id)
        return engine
    except Exception as e:
        logging.error(f"Failed to initialize TTS engine: {e}. Text-to-speech will be disabled.")
        return None  # Return None if TTS fails

# Initialize speech recognizer (only needed for voice mode)
def initialize_recognizer(config):
    """Initializes the speech recognition engine."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = config.get('energy_threshold', 300)
    recognizer.pause_threshold = config.get('pause_threshold', 0.8)
    # recognizer.dynamic_energy_threshold = True  # Optional: Can adapt to changing noise levels
    logging.info("Speech recognizer initialized.")
    return recognizer

# Queued TTS implementation
def _tts_worker(engine: pyttsx3.Engine | None, speaking_state: dict):
    """Worker thread to process TTS requests serially from a queue."""
    global engine_lock, speaking_state_lock  # Access global locks
    if engine is None:
        logging.warning("TTS engine not available. TTS worker thread exiting.")
        return
    base_rate = engine.getProperty("rate")
    while True:
        text = tts_queue.get()
        if text is None:  # Shutdown sentinel
            logging.info("TTS worker received shutdown signal.")
            tts_queue.task_done()
            break
        filtered_text = filter_text_for_speech(text)
        if not filtered_text:
            tts_queue.task_done()
            continue
        interrupted_during_speak = False
        # Set active = True *before* starting speech process
        with speaking_state_lock:
            speaking_state["active"] = True
            speaking_state["interrupted"] = False  # Reset interruption for this utterance
        # Reset engine state before speaking
        with engine_lock:
            try:
                engine.stop()  # Ensure any previous run is stopped
                if hasattr(engine, '_inLoop') and engine._inLoop:
                    engine.endLoop()
            except Exception as e:
                logging.warning(f"Minor issue resetting TTS loop: {e}")
        engine.setProperty("rate", base_rate + random.randint(-40, -10))
        try:
            # Lock engine during the blocking call
            with engine_lock:
                # Check interruption status *within the engine lock* right before runAndWait
                with speaking_state_lock:
                    if speaking_state["interrupted"]:
                        interrupted_during_speak = True
                        logging.info("TTS worker detected interruption just before speaking.")
                if not interrupted_during_speak:
                    logging.debug(f"TTS Engine saying: '{filtered_text[:50]}...'")
                    engine.say(filtered_text)
                    engine.runAndWait()  # This blocks until speech is done or stop() is called
                    logging.debug("TTS runAndWait finished.")
                # Attempt to clean up loop state after runAndWait finishes or if skipped
                try:
                    if hasattr(engine, '_inLoop') and engine._inLoop:
                        engine.endLoop()
                except Exception:
                    pass  # Ignore cleanup errors
        except RuntimeError as err:
            logging.error(f"pyttsx3 runtime error during speak: {err}; skipping.")
            with engine_lock:
                if hasattr(engine, '_inLoop') and engine._inLoop:
                    try:
                        engine.endLoop()
                    except Exception:
                        pass
        except Exception as e:
            logging.error(f"Unexpected error during TTS playback: {e}")
        finally:
            # --- Update State AFTER speech attempt ---
            with speaking_state_lock:
                # Lock before updating shared state
                engine.setProperty("rate", base_rate)  # Reset rate
                speaking_state["active"] = False  # Mark as no longer speaking
                if not speaking_state["interrupted"]:
                    speaking_state["last_end"] = time.time()
                speaking_state["interrupted"] = False  # Reset for next potential interruption
            tts_queue.task_done()  # Signal item processed
        # Optional pause only if not interrupted
        with speaking_state_lock:
            interrupted_flag_final = speaking_state["interrupted"]
        if not interrupted_flag_final:
            time.sleep(random.uniform(0.05, 0.15))

def speak_text(engine, text, speaking_state):
    """Adds text to the TTS queue for the worker to speak."""
    if engine:
        logging.debug(f"Queueing TTS: '{text[:50]}...'")
        tts_queue.put(text)
    else:
        logging.warning("TTS is disabled, cannot speak text.")

# Filter code blocks from text for speaking
def filter_text_for_speech(text):
    """Removes code blocks, URLs, etc., from text before speaking."""
    # Remove markdown code blocks first to avoid filtering content inside them
    text = re.sub(r'```[\s\S]*?```', ' (code block) ', text)
    # Remove inline code blocks
    text = re.sub(r'`[^`]*`', ' (code snippet) ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', ' (URL) ', text)
    # Replace special characters that might cause issues or are markdown
    text = text.replace("*", "").replace("_", "").replace("#", "")
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text.strip()

# Add human-like hesitation
def add_hesitation():
    """Adds a random hesitation phrase occasionally."""
    hesitations = ["Hmm...", "Let me think...", "Well...", "Okay...", "So..."]
    return random.choice(hesitations) + " " if random.random() < 0.2 else ""

# Try to import pywhatkit, but make it optional
try:
    import pywhatkit
    PYWHATKIT_AVAILABLE = True
except Exception as e:
    logging.warning(f"pywhatkit import failed: {e}. YouTube functionality disabled.")
    PYWHATKIT_AVAILABLE = False

# Search and play YouTube song
def search_and_play_youtube_song(query):
    """Searches and plays a YouTube video based on the query."""
    try:
        if PYWHATKIT_AVAILABLE:
            pywhatkit.playonyt(query)
            logging.info(f"Attempting to play '{query}' on YouTube via pywhatkit.")
            return f"Playing '{query}' on YouTube."
        else:
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"  # Standard search URL
            logging.info(f"pywhatkit not available. Opening YouTube search URL: {search_url}")
            if os.name == 'nt':
                os.system(f'start "{search_url}"')  # Quote URL for start command
            elif os.name == 'posix':
                os.system(f'open "{search_url}"' if os.uname().sysname == 'Darwin' else f'xdg-open "{search_url}"')  # Quote URL
            return f"I've opened a YouTube search for '{query}' in your browser."
    except Exception as e:
        logging.error(f"Error trying to play YouTube content for '{query}': {e}")
        return f"Sorry, I encountered an error trying to play '{query}' on YouTube."

# Detect programming language
def detect_language(code_content, code_block_language="", filename=""):
    """Detects the programming language from code content, hint, or filename."""
    # 1. Check filename extension first
    if filename:
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        ext_map = {
            ".py": ("python", ".py"),
            ".js": ("javascript", ".js"),
            ".java": ("java", ".java"),
            ".html": ("html", ".html"),
            ".htm": ("html", ".html"),
            ".css": ("css", ".css"),
            ".cpp": ("cpp", ".cpp"),
            ".cxx": ("cpp", ".cpp"),
            ".cc": ("cpp", ".cpp"),
            ".hpp": ("cpp", ".hpp"),
            ".c": ("c", ".c"),
            ".h": ("c", ".h"),
            ".ts": ("typescript", ".ts"),
            ".php": ("php", ".php"),
            ".go": ("go", ".go"),
            ".rb": ("ruby", ".rb"),
            ".swift": ("swift", ".swift"),
            ".sh": ("bash", ".sh"),
            ".bash": ("bash", ".sh"),
            ".sql": ("sql", ".sql"),
            ".json": ("json", ".json"),
            ".yaml": ("yaml", ".yaml"),
            ".yml": ("yaml", ".yaml"),
            ".md": ("markdown", ".md"),
            ".txt": ("text", ".txt")
        }
        if ext in ext_map:
            return ext_map[ext]
    # 2. Check markdown language hint
    if code_block_language and code_block_language.strip():
        lang = code_block_language.strip().lower()
        hint_map = {
            "python": ("python", ".py"),
            "py": ("python", ".py"),
            "javascript": ("javascript", ".js"),
            "js": ("javascript", ".js"),
            "java": ("java", ".java"),
            "html": ("html", ".html"),
            "css": ("css", ".css"),
            "cpp": ("cpp", ".cpp"),
            "c++": ("cpp", ".cpp"),
            "c": ("c", ".c"),
            "typescript": ("typescript", ".ts"),
            "ts": ("typescript", ".ts"),
            "php": ("php", ".php"),
            "go": ("go", ".go"),
            "ruby": ("ruby", ".rb"),
            "rb": ("ruby", ".rb"),
            "swift": ("swift", ".swift"),
            "bash": ("bash", ".sh"),
            "sh": ("bash", ".sh"),
            "sql": ("sql", ".sql"),
            "json": ("json", ".json"),
            "yaml": ("yaml", ".yaml"),
            "yml": ("yaml", ".yaml"),
            "markdown": ("markdown", ".md"),
            "md": ("markdown", ".md"),
        }
        if lang in hint_map:
            return hint_map[lang]
        else:
            logging.info(f"Unknown language hint '{lang}'. Attempting content detection.")
    # 3. Content-based detection (simplified fallback)
    if code_content:
        # Only check content if available
        code_lower = code_content.lower()
        if "import " in code_lower and ("def " in code_lower or "class " in code_lower):
            return "python", ".py"
        if "function" in code_lower and ("var " in code_lower or "const " in code_lower or "let " in code_lower):
            return "javascript", ".js"
        if "public class" in code_lower or "system.out.println" in code_lower:
            return "java", ".java"
        if "<!doctype html>" in code_lower or "<html>" in code_lower:
            return "html", ".html"
        if re.search(r'\{\s*[\w-]+\s*:', code_content):
            return "css", ".css"
        if "#include" in code_lower and ("int main" in code_lower or "std::cout" in code_lower):
            return "cpp", ".cpp"
        if "<?php" in code_lower:
            return "php", ".php"
        if "package main" in code_lower and "func main" in code_lower:
            return "go", ".go"
        if "require " in code_lower and " end" in code_lower:
            return "ruby", ".rb"
        if "import swift" in code_lower or "func " in code_lower:
            return "swift", ".swift"
        if "select " in code_lower and " from " in code_lower:
            return "sql", ".sql"
        if code_content.strip().startswith("{") and code_content.strip().endswith("}"):
            return "json", ".json"
        if re.match(r'^[\w-]+:', code_content.strip(), re.MULTILINE):
            return "yaml", ".yaml"
    logging.info("Could not reliably detect language. Defaulting to 'text', '.txt'.")
    return "text", ".txt"

# Create and save code files
def create_code_file(filename, code_content, language_hint=""):
    """Creates a file with the given code content and correct extension."""
    try:
        # Detect language using content and hint, also pass filename for extension check
        lang, extension = detect_language(code_content, language_hint, filename)
        logging.info(f"Detected language: {lang}, using extension: {extension}")
        name, ext = os.path.splitext(filename)
        # Ensure filename has the *detected* correct extension
        if ext.lower() != extension.lower():
            filename = name + extension if ext else filename + extension
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code_content)
        logging.info(f"Successfully created file: {filename}")
        return True, filename, lang
    except Exception as e:
        logging.error(f"Error creating file '{filename}': {e}")
        return False, filename, None

# Open file for editing
def open_file_for_editing(filename):
    """Opens a file in the system's default application."""
    try:
        if not os.path.exists(filename):
            logging.error(f"Cannot open file: '{filename}' does not exist.")
            return False
        logging.info(f"Attempting to open file '{filename}' for editing.")
        if os.name == 'nt':
            os.startfile(filename)
        elif os.name == 'posix':
            cmd = 'open' if os.uname().sysname == 'Darwin' else 'xdg-open'
            subprocess.run([cmd, filename], check=True)
        else:
            logging.warning(f"Unsupported OS '{os.name}'. Cannot automatically open file.")
            return False
        return True
    except FileNotFoundError:
        logging.error(f"Could not find command ('open' or 'xdg-open') to open the file.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Command to open file failed: {e}")
        return False
    except Exception as e:
        logging.error(f"Error opening file '{filename}': {e}")
        return False

# Get current date and time
def get_current_datetime():
    """Returns a formatted string of the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("It's %A, %B %d, %Y, and the time is %I:%M %p.")

# Run shell commands securely
def run_shell_command(command, is_execution=False):
    """Executes a shell command using subprocess and returns output. Includes safety checks for allowed commands.
    `is_execution` flag indicates if this is running a user script file."""
    # ** Ensure python, python3, node are allowed for execution **
    allowed_commands = ["npx", "npm", "pip", "git", "python", "python3", "node", "javac", "java", "echo", "ls", "dir", "cat", "type"]
    try:
        # For execution, command might be just "python file.py"
        # For other commands, split normally
        if is_execution:
            command_parts = command.split(maxsplit=1)  # e.g., ['python', 'file.py args']
        else:
            command_parts = command.split()
        if not command_parts:
            return "No command provided.", "No command provided."
        executable = command_parts[0].lower()
        if executable not in allowed_commands:
            logging.warning(f"Blocked potentially unsafe command: {command}")
            msg = f"Sorry, I can only run specific safe commands. Cannot run '{command_parts[0]}'."
            return msg, msg
        logging.info(f"Executing command: {command}")
        # Use shell=True for convenience, but be aware of security implications
        # Setting cwd might be useful if scripts expect to run from their own directory
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False, timeout=180)  # Increased timeout for execution
        # Combine stdout and stderr for the detailed response, limit length
        full_output = (result.stdout or "") + ("\n--- Errors ---\n" + result.stderr if result.stderr else "")
        output_snippet = full_output[:1500]  # Slightly larger snippet for execution output
        output_message = f"Command: `{command}`\nReturn Code: {result.returncode}\n"
        if output_snippet:
            output_message += f"Output:\n```\n{output_snippet}\n```\n"
        elif result.returncode != 0:
            output_message += "(No output captured, but command failed)\n"
        if result.returncode == 0:
            logging.info(f"Command '{command}' executed successfully.")
            if is_execution:
                spoken_confirmation = f"Execution of `{command}` finished successfully."
            elif executable in ["cat", "type"]:
                spoken_confirmation = f"Displayed content of {command_parts[1] if len(command_parts) > 1 else 'file'}."
            else:
                spoken_confirmation = f"Command `{command}` executed successfully."
        else:
            logging.error(f"Command '{command}' failed. Error: {result.stderr}")
            spoken_confirmation = f"Command `{command}` failed."
            if is_execution:
                spoken_confirmation = f"Execution of `{command}` failed."
        return output_message, spoken_confirmation
    except subprocess.TimeoutExpired:
        logging.error(f"Command '{command}' timed out.")
        msg = f"Command `{command}` timed out."
        return msg, msg
    except FileNotFoundError:
        logging.error(f"Executable '{command_parts[0]}' not found.")
        msg = f"Error: Command '{command_parts[0]}' not found."
        return msg, msg
    except Exception as e:
        logging.error(f"Error running command '{command}': {e}")
        msg = f"An error occurred running the command: {e}"
        return msg, "Command error."

# --- File Modification Function (with enhanced logging) ---
def modify_file_content(filename, instruction):
    """Modifies the content of a file based on a simple instruction.
    Supported instructions:
    - append <text>
    - prepend <text>
    - replace "<find_text>" with "<replace_text>" (first occurrence, uses quotes)
    - delete line <number>"""
    logging.info(f"Attempting to modify file '{filename}' with instruction: '{instruction}'")
    if not os.path.exists(filename):
        logging.error(f"File modification failed: '{filename}' not found.")
        return f"Error: File '{filename}' not found.", False
    try:
        # Read current content
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # Read as lines for line deletion
        modified = False
        original_content = "".join(lines)
        new_content = original_content  # Start with original
        action_desc = ""  # Initialize action description
        # Parse instruction (case-insensitive matching for keywords)
        instruction_lower = instruction.lower()
        # Append
        match_append = re.match(r'append\s+(.*)', instruction, re.IGNORECASE | re.DOTALL)
        if match_append:
            logging.debug("Modification type: append")
            text_to_append = match_append.group(1).strip()
            if text_to_append.startswith(('"', "'")) and text_to_append.endswith(('"', "'")):
                text_to_append = text_to_append[1:-1]
            if new_content and not new_content.endswith('\n'):
                new_content += '\n'
            new_content += text_to_append
            modified = True
            action_desc = f"appended text to '{filename}'"
        # Prepend
        match_prepend = re.match(r'prepend\s+(.*)', instruction, re.IGNORECASE | re.DOTALL)
        if not modified and match_prepend:
            logging.debug("Modification type: prepend")
            text_to_prepend = match_prepend.group(1).strip()
            if text_to_prepend.startswith(('"', "'")) and text_to_prepend.endswith(('"', "'")):
                text_to_prepend = text_to_prepend[1:-1]
            if new_content and not new_content.startswith('\n'):
                new_content = text_to_prepend + '\n' + new_content
            else:
                new_content = text_to_prepend + new_content
            modified = True
            action_desc = f"prepended text to '{filename}'"
        # Replace (first occurrence) - Requires quotes around find/replace text
        match_replace = re.match(r'replace\s+["\'](.*?)["\']\s+with\s+["\'](.*?)["\']', instruction, re.IGNORECASE | re.DOTALL)
        if not modified and match_replace:
            logging.debug("Modification type: replace")
            find_text = match_replace.group(1)
            replace_text = match_replace.group(2)
            if find_text in new_content:
                new_content = new_content.replace(find_text, replace_text, 1)
                modified = True
                action_desc = f"replaced '{find_text}' in '{filename}'"
            else:
                logging.warning(f"Replace failed: '{find_text}' not found in '{filename}'.")
                return f"Could not find '{find_text}' in '{filename}'.", False
        # Delete line
        match_delete = re.match(r'delete line\s+(\d+)', instruction, re.IGNORECASE)
        if not modified and match_delete:
            logging.debug("Modification type: delete line")
            try:
                line_num_to_delete = int(match_delete.group(1))
                if 1 <= line_num_to_delete <= len(lines):
                    del lines[line_num_to_delete - 1]  # Adjust for 0-based index
                    new_content = "".join(lines)
                    modified = True
                    action_desc = f"deleted line {line_num_to_delete} from '{filename}'"
                else:
                    logging.warning(f"Delete line failed: Invalid line number {line_num_to_delete} for file '{filename}' ({len(lines)} lines).")
                    return f"Invalid line number {line_num_to_delete} for file '{filename}'.", False
            except ValueError:
                logging.warning("Delete line failed: Invalid number format.")
                return "Invalid line number provided for deletion.", False
        # If modification occurred, write back to file
        if modified:
            logging.info(f"Modification successful ({action_desc}). Attempting to write changes to '{filename}'.")
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logging.info(f"Successfully wrote modified content to '{filename}'.")
                return f"Okay, I {action_desc}.", True
            except Exception as write_err:
                logging.error(f"Error writing modified content to '{filename}': {write_err}")
                return f"An error occurred while writing changes to '{filename}'.", False
        else:
            # If no pattern matched
            logging.warning(f"No modification pattern matched for instruction: '{instruction}'")
            return f"Sorry, I couldn't parse that modification instruction. Please use 'append [text]', 'prepend [text]', 'replace \"[find]\" with \"[replace]\"', or 'delete line [number]'.", False
    except Exception as e:
        logging.error(f"Error modifying file '{filename}': {e}", exc_info=True)  # Log traceback
        return f"An unexpected error occurred while trying to modify '{filename}'.", False

# --- Show File Content Function ---
def show_file_content(filename, max_lines=50):
    """Reads and returns the content of a file, limited by max_lines."""
    if not os.path.exists(filename):
        return f"Error: File '{filename}' not found.", False
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        num_lines = len(lines)
        # Handle potential encoding errors during read/join if needed
        content_snippet = "".join(lines[:max_lines])
        result_message = f"Content of '{filename}' ({num_lines} lines total):\n```\n{content_snippet}\n```"
        if num_lines > max_lines:
            result_message += f"\n(Showing first {max_lines} lines)"
        logging.info(f"Read content from '{filename}'.")
        return result_message, True
    except Exception as e:
        logging.error(f"Error reading file '{filename}': {e}")
        return f"An error occurred while trying to read '{filename}'.", False

# Read full file content
def read_full_file(filename):
    """Reads the full content of a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        logging.info(f"Read full content from '{filename}'.")
        return content, False
    except Exception as e:
        logging.error(f"Error reading file '{filename}': {e}")
        return f"An error occurred while trying to read '{filename}'.", True

# Extract code from response
def extract_code_from_response(response):
    """Extracts code from a response."""
    try:
        code_blocks = re.findall(r'```([\w+-]*)\n?([\s\S]*?)```', response, re.DOTALL)
        if code_blocks:
            # Return the first code block
            return code_blocks[0][1].strip()
        else:
            logging.warning("No code blocks found in response.")
            return None
    except Exception as e:
        logging.error(f"Error extracting code from response: {e}")
        return None

# --- Text-based Input Loop ---
def text_based_chat(chain, engine, config):
    """Handles interaction using text input instead of speech recognition."""
    global is_speaking, speaking_state_lock  # Access global state
    conversation_history = []
    # Start TTS worker thread if engine is available
    tts_thread = None
    if engine:
        tts_thread = threading.Thread(target=_tts_worker, args=(engine, is_speaking), name="TTSWorkerThread", daemon=True)
        tts_thread.start()
    # Initial greeting
    initial_greeting = random.choice(config.get('greeting_phrases', ["Hello!"])) + " (Text Mode)"
    logging.info(f"Sam: {initial_greeting}")
    print(f"\nSam: {initial_greeting}")
    speak_text(engine, initial_greeting, is_speaking)
    running = True
    while running:
        try:
            # Get text input from the user
            print("\nYou: ", end="")
            user_input = input().strip()
            if not user_input:
                continue  # Ignore empty input
            logging.info(f"Text input received: '{user_input}'")
            response = ""
            spoken_response = ""
            user_input_lower = user_input.lower()  # Pre-calculate lower case
            # --- Process Text Input (Shared Logic) ---
            # This block contains the command processing logic, shared between text and voice modes
            # --- Exit Condition ---
            if any(p in user_input_lower for p in ['goodbye', 'bye', 'exit', 'quit', 'shutdown', 'turn off']):
                spoken_response = random.choice(config.get('goodbye_phrases', ["Goodbye!"]))
                running = False  # Process response before exiting loop
            # --- NEW: Run File ---
            elif re.match(r'(run|execute)\s+(?:script|file)?\s*([\w.-]+)', user_input, re.IGNORECASE):
                match = re.match(r'(run|execute)\s+(?:script|file)?\s*([\w.-]+)', user_input, re.IGNORECASE)
                filename_to_run = match.group(2)
                lang, _ = detect_language("", "", filename_to_run)  # Detect language from filename extension
                command_to_run_script = None
                if lang == "python":
                    # Try python3 first, then python
                    command_to_run_script = f"python3 {filename_to_run}"
                    # Simple check if python3 exists (optional, adds complexity)
                    # try: subprocess.run(["python3", "--version"], check=True, capture_output=True)
                    # except (FileNotFoundError, subprocess.CalledProcessError):
                    #     command_to_run_script = f"python {filename_to_run}"
                elif lang == "javascript":
                    command_to_run_script = f"node {filename_to_run}"
                # Add other languages here if needed (e.g., java requires compile step)
                if command_to_run_script:
                    response, spoken_response = run_shell_command(command_to_run_script, is_execution=True)
                else:
                    response = f"Sorry, I don't know how to run '{filename_to_run}' (unsupported language: {lang}). I can run .py and .js files."
                    spoken_response = response
            # --- Show File Content ---
            elif re.match(r'(show|display|view)\s+(?:content of\s+)?(?:file\s+)?([\w.-]+)', user_input, re.IGNORECASE):
                match = re.match(r'(show|display|view)\s+(?:content of\s+)?(?:file\s+)?([\w.-]+)', user_input, re.IGNORECASE)
                filename_to_show = match.group(2)
                response, success = show_file_content(filename_to_show)
                spoken_response = f"Showing content of {filename_to_show}." if success else response
            # --- Modify File Content (Improved Regex) ---
            # Pattern 1: modify file <filename> <instruction>
            elif re.match(r'(modify|change|edit)\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL):
                match = re.match(r'(modify|change|edit)\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL)
                filename_to_modify = match.group(2)
                modification_instruction = match.group(3).strip()
                spoken_response, success = modify_file_content(filename_to_modify, modification_instruction)
                response = spoken_response  # Use same for print
            # Pattern 2: in file <filename> <instruction>
            elif re.match(r'in\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL):
                match = re.match(r'in\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL)
                filename_to_modify = match.group(1)
                modification_instruction = match.group(2).strip()
                spoken_response, success = modify_file_content(filename_to_modify, modification_instruction)
                response = spoken_response  # Use same for print
            # --- Weather Query ---
            elif re.match(r"what'?s the weather(?: like)?\s+(?:in|for)\s+(.+)", user_input, re.IGNORECASE):
                match = re.match(r"what'?s the weather(?: like)?\s+(?:in|for)\s+(.+)", user_input, re.IGNORECASE)
                location = match.group(1).strip()
                weather_query = f"What is the current weather like in {location}?"
                logging.info(f"Passing weather query to LLM: {weather_query}")
                response = add_hesitation() + chain.run(conversation_history=" ".join(conversation_history[-10:]), user_input=weather_query)
                spoken_response = filter_text_for_speech(response)
            # --- YouTube Request ---
            elif ("play" in user_input_lower or "search for" in user_input_lower) and \
                 ("on youtube" in user_input_lower or "youtube song" in user_input_lower):
                match = re.search(r'(?:play|search for)\s+(.*?)\s+(?:on youtube|youtube song)', user_input, re.IGNORECASE)
                if match:
                    query = match.group(1).strip()
                    spoken_response = search_and_play_youtube_song(query)
                else:
                    spoken_response = "What would you like me to play or search for on YouTube?"
                response = spoken_response
            # --- Date/Time Request ---
            elif any(w in user_input_lower for w in ["date", "time", "day", "what's the date", "what time is it"]):
                spoken_response = get_current_datetime()
                response = spoken_response
            # --- Shell Command Execution Request ---
            elif user_input_lower.startswith(("run ", "execute ", "perform ")) and not \
                 (user_input_lower.startswith(("run cat ", "run type ", "execute cat ", "execute type "))):
                # Avoid conflict with show
                # Also avoid conflict with run file
                if not re.match(r'(run|execute)\s+(?:script|file)?\s*([\w.-]+)', user_input, re.IGNORECASE):
                    # Ensure there's actually a command after "run " etc.
                    command_parts = user_input.split(maxsplit=1)
                    if len(command_parts) > 1:
                        command_to_run = command_parts[1]
                        response, spoken_response = run_shell_command(command_to_run, is_execution=False)
                    else:
                        response = spoken_response = "Please specify a command to run."
                # else: handled by "run file" case above
            # --- Code Generation & File Handling ---
            elif any(t in user_input_lower for t in ["create", "make", "build", "generate", "write", "give me"]) and \
                 any(t in user_input_lower for t in ["code", "script", "program", "function", "class", "file", "app", "html", "python", "javascript"]):
                llm_response = add_hesitation() + chain.run(conversation_history=" ".join(conversation_history[-10:]), user_input=user_input)
                response = llm_response
                code_blocks = re.findall(r'```([\w+-]*)\n?([\s\S]*?)```', llm_response, re.DOTALL)
                if code_blocks:
                    summary = []
                    saved = 0
                    opened = 0
                    for lang_hint, code in code_blocks:
                        code = code.strip()
                        if not code:
                            continue
                        fname_match = re.search(r'(?:file|script|app)\s+(?:named|called)\s+([\w.-]+)', user_input, re.IGNORECASE)
                        fname = fname_match.group(1) if fname_match else f"generated_code_{int(time.time()) % 1000}"
                        ok, final_fname, lang = create_code_file(fname, code, lang_hint)
                        if ok:
                            saved += 1
                            summary.append(f"Saved '{final_fname}' ({lang}).")
                            if re.search(r'(open|edit)\s+(it|the file)', user_input_lower) or \
                               (fname_match and re.search(f'(open|edit)\s+{re.escape(fname)}', user_input_lower, re.IGNORECASE)):
                                if open_file_for_editing(final_fname):
                                    opened += 1
                                    summary.append(f" Opened '{final_fname}'.")
                                else:
                                    summary.append(f" Couldn't open '{final_fname}'.")
                    spk = "Okay, generated code. "
                    if saved:
                        spk += f"Saved {saved} file(s). "
                    if opened:
                        spk += f"Opened {opened}. "
                    elif saved:
                        spk += "Ask to open if needed."
                    else:
                        spk = "Generated code, but failed to save."
                    spoken_response = spk
                    if summary:
                        response += "\n\n--- File Summary ---\n" + "\n".join(summary)
                else:
                    # No code blocks found
                    spoken_response = filter_text_for_speech(llm_response) or "I tried, but couldn't generate that."
            # --- File Editing Request (Standalone - Open Only) ---
            elif re.match(r'(open)\s+(?:file\s+)?([\w.-]+)', user_input, re.IGNORECASE) and \
                 not (re.match(r'(modify|change|edit)\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE) or \
                      re.match(r'in\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE)):
                # Avoid conflict with modify
                match = re.match(r'(open|edit)\s+(?:file\s+)?([\w.-]+)', user_input, re.IGNORECASE)
                fname_to_edit = match.group(2)
                if open_file_for_editing(fname_to_edit):
                    spoken_response = f"Opened '{fname_to_edit}'."
                else:
                    spoken_response = f"Sorry, couldn't open '{fname_to_edit}'."
                response = spoken_response
            # --- AI Edit File (LLM-powered) ---
            elif re.match(r'(?:edit|modify|update) (?:this|it|file)\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL):
                match = re.match(r'(?:edit|modify) (?:this|it|file)\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL)
                filename_to_edit = match.group(1)
                edit_instruction = match.group(2).strip()
                logging.info(f"Starting AI edit for '{filename_to_edit}'. Instruction: '{edit_instruction}'")
                original_code, read_error = read_full_file(filename_to_edit)
                if read_error:
                    response = spoken_response = read_error
                else:
                    ai_edit_prompt = f"""You are an expert programmer AI. Below is the content of the file '{filename_to_edit}' and a user request to modify it.\nUser Request: '{edit_instruction}'\n\nCurrent File Content:\n``` {original_code} ```\n\nYour task is to apply the user's request to the code. Return ONLY the complete, modified code for the entire file, enclosed in a single markdown code block (```). Do not add explanations before or after the code block. Ensure the modified code is syntactically correct and functional based on the request. If the request cannot be fulfilled reasonably, return the original code block unchanged.\n"""
                    logging.debug(f"Sending prompt to LLM for AI Edit:\n{ai_edit_prompt}")
                    try:
                        logging.info("Sending AI edit request to LLM...")
                        ai_response = chain.run(conversation_history="", user_input=ai_edit_prompt)
                        logging.debug(f"LLM raw response for AI edit:\n{ai_response}")
                        proposed_code = extract_code_from_response(ai_response)
                        if proposed_code:
                            if proposed_code.strip() == original_code.strip():
                                response = "The AI didn't propose any changes based on your instruction."
                                spoken_response = "The AI didn't propose any changes."
                            else:
                                try:
                                    with open(filename_to_edit, 'w', encoding='utf-8') as f:
                                        f.write(proposed_code)
                                    response = f"AI edit complete! Changes applied to '{filename_to_edit}'."
                                    spoken_response = f"AI edit complete. Changes saved to {filename_to_edit}."
                                except Exception as e:
                                    response = f"Sorry, failed to save changes to '{filename_to_edit}': {e}"
                                    spoken_response = response
                        else:
                            response = "Sorry, I couldn't extract the modified code from the AI response. No changes made."
                            spoken_response = "Sorry, I had trouble generating the changes."
                    except Exception as llm_err:
                        logging.error(f"Error during LLM call for AI edit: {llm_err}", exc_info=True)
                        response = "Sorry, an error occurred while communicating with the AI for editing."
                        spoken_response = response
            # --- Default: General Conversation ---
            else:
                llm_response = add_hesitation() + chain.run(conversation_history=" ".join(conversation_history[-10:]), user_input=user_input)
                response = llm_response
                spoken_response = filter_text_for_speech(llm_response)
            # --- Process and Speak Response ---
            if response:
                logging.info(f"Sam: {response}")  # Log full response
                print(f"\nSam: {response}")  # Print full response
                if spoken_response:
                    speak_text(engine, spoken_response, is_speaking)
                else:
                    logging.warning("Filtered response for speech is empty.")
            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Sam: {response}")
            conversation_history = conversation_history[-20:]  # Limit history
            # If exit command was processed, exit loop now
            if not running:
                time.sleep(2)  # Allow TTS to potentially finish
                break  # Exit the while loop
        except EOFError:
            # Handle Ctrl+D
            print("\nEOF received. Exiting text mode.")
            running = False
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nKeyboard interrupt received. Exiting text mode.")
            running = False
        except Exception as e:
            logging.error(f"Error in text chat loop: {e}", exc_info=True)
            error_msg = "Oops, an error occurred in text mode. Please try again."
            print(f"\nSam: {error_msg}")
            try:
                speak_text(engine, error_msg, is_speaking)
            except Exception:
                pass
            time.sleep(1)
    # --- Cleanup for Text Mode ---
    logging.info("Shutting down text-based chat...")
    if tts_thread:
        logging.info("Signaling TTS worker to stop...")
        tts_queue.put(None)
        tts_queue.join()
        tts_thread.join(timeout=5)
        if tts_thread.is_alive():
            logging.warning("TTS thread did not exit cleanly.")
    if engine:
        try:
            with engine_lock:
                if hasattr(engine, '_inLoop') and engine._inLoop:
                    engine.endLoop()
        except Exception as e:
            logging.warning(f"Final TTS cleanup error: {e}")
    logging.info("Sam AI Assistant (Text Mode) shut down.")
    print("\nSam AI Assistant (Text Mode) shut down.")

# --- Voice Chat Logic ---
def start_chat(chain, engine, recognizer, config):
    """Main loop for handling conversation using voice recognition."""
    global is_speaking, speaking_state_lock, command_queue  # Access globals
    conversation_history = []
    listener_stop_func = None  # Initialize variable to hold the stop function
    # Start TTS worker thread if engine is available
    tts_thread = None
    if engine:
        tts_thread = threading.Thread(target=_tts_worker, args=(engine, is_speaking), name="TTSWorkerThread", daemon=True)
        tts_thread.start()
    # --- Background Speech Recognition Callback ---
    def audio_callback(rec, audio):
        """Processes audio captured by the background listener."""
        global is_speaking, speaking_state_lock, command_queue, engine_lock  # Access globals
        with speaking_state_lock:
            currently_speaking = is_speaking.get('active', False)
            last_speak_end_time = is_speaking.get("last_end", 0)
            is_waiting_for_cmd = is_speaking.get("waiting_for_command", False)
            min_listen_delay = 0.8  # Listen more eagerly after speaking
        if currently_speaking:
            # Allow only interruption keywords while TTS is speaking
            try:
                phrase_inter = rec.recognize_google(audio, language="en-US")
                lower_inter = phrase_inter.lower()
                stop_keywords = ["stop", "quiet", "shh", "be quiet", "shut up", "enough"]
                if any(k in lower_inter for k in stop_keywords):
                    print(f"\n>>> You interrupted with: {phrase_inter}")
                    logging.info(f"Interruption detected: '{phrase_inter}'")
                    with speaking_state_lock:
                        is_speaking["interrupted"] = True
                    if engine:
                        with engine_lock:
                            engine.stop()
                            # Flush pending TTS queue to stop any queued utterances
                            while not tts_queue.empty():
                                try:
                                    tts_queue.get_nowait()
                                    tts_queue.task_done()
                                except queue.Empty:
                                    break
            except (sr.UnknownValueError, sr.RequestError):
                pass
            except Exception as e:
                logging.error(f"Error during interruption check: {e}")
            finally:
                return  # Do not process further audio while speaking
        time_since_last_speak = time.time() - last_speak_end_time
        if time_since_last_speak < min_listen_delay:
            return
        print("Listening...")
        try:
            phrase = rec.recognize_google(audio, language="en-US")
            logging.debug(f"SR recognized: '{phrase}'")
            print(f"\n>>> You said: {phrase}")
            lower_phrase = phrase.lower()
            # Expanded wake words slightly
            wake_words = ["sam", "hey sam", "okay sam", "hello sam", "hey", "alright"]
            is_wake_word = any(lower_phrase.startswith(w) for w in wake_words)
            if is_wake_word:
                logging.info(f"Wake word detected: '{phrase}'")
                command_queue.put(("wake", phrase))
                with speaking_state_lock:
                    is_speaking["waiting_for_command"] = True
            elif is_waiting_for_cmd or not is_wake_word:
                logging.info(f"Command received: '{phrase}'")
                command_queue.put(("command", phrase))
                with speaking_state_lock:
                    is_speaking["waiting_for_command"] = False
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            logging.error(f"SR service error; {e}")
        except Exception as e:
            logging.error(f"Error processing recognized phrase: {e}")
    # --- Start Background Listener ---
    try:
        mic = sr.Microphone()
        with mic as source:
            logging.info("Adjusting for ambient noise...")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
            except Exception as adjust_err:
                logging.warning(f"Ambient noise adjustment failed: {adjust_err}")
            logging.info(f"Ambient noise adjustment complete. Threshold: {recognizer.energy_threshold:.2f}")
            listener_stop_func = recognizer.listen_in_background(mic, audio_callback, phrase_time_limit=5)
            logging.info("Background listener started.")
    except Exception as e:
        logging.error(f"Failed to start background listener: {e}. Voice input disabled.")
    # --- Initial Greeting ---
    initial_greeting = random.choice(config.get('greeting_phrases', ["Hello!"]))
    logging.info(f"Sam: {initial_greeting}")
    print(f"Sam: {initial_greeting}")
    speak_text(engine, initial_greeting, is_speaking)
    # --- Main Interaction Loop (Voice Mode) ---
    running = True
    while running:
        try:
            cmd_type, user_input = command_queue.get(timeout=0.1)  # Check queue from callback
            logging.debug(f"Processing command: Type={cmd_type}, Input='{user_input}'")
            response = ""
            spoken_response = ""
            user_input_lower = user_input.lower()  # Pre-calculate lower case
            # --- Handle Wake Word ---
            if cmd_type == "wake":
                spoken_response = random.choice(["Yes?", "I'm here.", "Listening."])
                logging.info(f"Sam: {spoken_response}")
                print(f"Sam: {spoken_response}")
                speak_text(engine, spoken_response, is_speaking)
                continue
            # --- Handle Actual Command ---
            if cmd_type == "command":
                with speaking_state_lock:
                    is_speaking["waiting_for_command"] = False
                # --- Process Commands (Shared Logic Block) ---
                # This block should be identical to the one in text_based_chat
                # --- Exit Condition ---
                if any(p in user_input_lower for p in ['goodbye sam', 'bye sam', 'exit', 'quit', 'shutdown', 'turn off']):
                    spoken_response = random.choice(config.get('goodbye_phrases', ["Goodbye!"]))
                    running = False  # Process response before exiting loop
                # --- NEW: Run File ---
                elif re.match(r'(run|execute)\s+(?:script|file)?\s*([\w.-]+)', user_input, re.IGNORECASE):
                    match = re.match(r'(run|execute)\s+(?:script|file)?\s*([\w.-]+)', user_input, re.IGNORECASE)
                    filename_to_run = match.group(2)
                    lang, _ = detect_language("", "", filename_to_run)
                    command_to_run_script = None
                    if lang == "python":
                        command_to_run_script = f"python3 {filename_to_run}"  # Or just python
                    elif lang == "javascript":
                        command_to_run_script = f"node {filename_to_run}"
                    if command_to_run_script:
                        response, spoken_response = run_shell_command(command_to_run_script, is_execution=True)
                    else:
                        response = f"Sorry, I don't know how to run '{filename_to_run}' (unsupported language: {lang})."
                        spoken_response = response
                # --- Show File Content ---
                elif re.match(r'(show|display|view)\s+(?:content of\s+)?(?:file\s+)?([\w.-]+)', user_input, re.IGNORECASE):
                    match = re.match(r'(show|display|view)\s+(?:content of\s+)?(?:file\s+)?([\w.-]+)', user_input, re.IGNORECASE)
                    filename_to_show = match.group(2)
                    response, success = show_file_content(filename_to_show)
                    spoken_response = f"Showing content of {filename_to_show}." if success else response
                # --- Modify File Content (Improved Regex) ---
                elif re.match(r'(modify|change|edit)\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL):
                    match = re.match(r'(modify|change|edit)\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL)
                    filename_to_modify = match.group(2)
                    modification_instruction = match.group(3).strip()
                    spoken_response, success = modify_file_content(filename_to_modify, modification_instruction)
                    response = spoken_response
                elif re.match(r'in\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL):
                    match = re.match(r'in\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE | re.DOTALL)
                    filename_to_modify = match.group(1)
                    modification_instruction = match.group(2).strip()
                    spoken_response, success = modify_file_content(filename_to_modify, modification_instruction)
                    response = spoken_response
                # --- Weather Query ---
                elif re.match(r"what'?s the weather(?: like)?\s+(?:in|for)\s+(.+)", user_input, re.IGNORECASE):
                    match = re.match(r"what'?s the weather(?: like)?\s+(?:in|for)\s+(.+)", user_input, re.IGNORECASE)
                    location = match.group(1).strip()
                    weather_query = f"What is the current weather like in {location}?"
                    logging.info(f"Passing weather query to LLM: {weather_query}")
                    response = add_hesitation() + chain.run(conversation_history=" ".join(conversation_history[-10:]), user_input=weather_query)
                    spoken_response = filter_text_for_speech(response)
                # --- YouTube Request ---
                elif ("play" in user_input_lower or "search for" in user_input_lower) and \
                     ("on youtube" in user_input_lower or "youtube song" in user_input_lower):
                    match = re.search(r'(?:play|search for)\s+(.*?)\s+(?:on youtube|youtube song)', user_input, re.IGNORECASE)
                    if match:
                        query = match.group(1).strip()
                        spoken_response = search_and_play_youtube_song(query)
                    else:
                        spoken_response = "What would you like me to play or search for on YouTube?"
                    response = spoken_response
                # --- Date/Time Request ---
                elif any(w in user_input_lower for w in ["date", "time", "day", "what's the date", "what time is it"]):
                    spoken_response = get_current_datetime()
                    response = spoken_response
                # --- Shell Command Execution Request ---
                elif user_input_lower.startswith(("run ", "execute ", "perform ")) and not \
                     (user_input_lower.startswith(("run cat ", "run type ", "execute cat ", "execute type "))):
                    # Avoid conflict with show
                    if not re.match(r'(run|execute)\s+(?:script|file)?\s*([\w.-]+)', user_input, re.IGNORECASE):
                        # Ensure there's actually a command after "run " etc.
                        command_parts = user_input.split(maxsplit=1)
                        if len(command_parts) > 1:
                            command_to_run = command_parts[1]
                            response, spoken_response = run_shell_command(command_to_run, is_execution=False)
                        else:
                            response = spoken_response = "Please specify a command to run."
                # --- Code Generation & File Handling ---
                elif any(t in user_input_lower for t in ["create", "make", "build", "generate", "write", "give me"]) and \
                     any(t in user_input_lower for t in ["code", "script", "program", "function", "class", "file", "app", "html", "python", "javascript"]):
                    llm_response = add_hesitation() + chain.run(conversation_history=" ".join(conversation_history[-10:]), user_input=user_input)
                    response = llm_response
                    code_blocks = re.findall(r'```([\w+-]*)\n?([\s\S]*?)```', llm_response, re.DOTALL)
                    if code_blocks:
                        summary = []
                        saved = 0
                        opened = 0
                        for lang_hint, code in code_blocks:
                            code = code.strip()
                            if not code:
                                continue
                            fname_match = re.search(r'(?:file|script|app)\s+(?:named|called)\s+([\w.-]+)', user_input, re.IGNORECASE)
                            fname = fname_match.group(1) if fname_match else f"generated_code_{int(time.time()) % 1000}"
                            ok, final_fname, lang = create_code_file(fname, code, lang_hint)
                            if ok:
                                saved += 1
                                summary.append(f"Saved '{final_fname}' ({lang}).")
                                if re.search(r'(open|edit)\s+(it|the file)', user_input_lower) or \
                                   (fname_match and re.search(f'(open|edit)\s+{re.escape(fname)}', user_input_lower, re.IGNORECASE)):
                                    if open_file_for_editing(final_fname):
                                        opened += 1
                                        summary.append(f" Opened '{final_fname}'.")
                                    else:
                                        summary.append(f" Couldn't open '{final_fname}'.")
                        spk = "Okay, generated code. "
                        if saved:
                            spk += f"Saved {saved} file(s). "
                        if opened:
                            spk += f"Opened {opened}. "
                        elif saved:
                            spk += "Ask to open if needed."
                        else:
                            spk = "Generated code, but failed to save."
                        spoken_response = spk
                        if summary:
                            response += "\n\n--- File Summary ---\n" + "\n".join(summary)
                    else:
                        spoken_response = filter_text_for_speech(llm_response) or "I tried, but couldn't generate that."
                # --- File Editing Request (Standalone - Open Only) ---
                elif re.match(r'(open|edit)\s+(?:file\s+)?([\w.-]+)', user_input, re.IGNORECASE) and \
                     not (re.match(r'(modify|change|edit)\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE) or \
                          re.match(r'in\s+file\s+([\w.-]+)\s+(.*)', user_input, re.IGNORECASE)):
                    match = re.match(r'(open|edit)\s+(?:file\s+)?([\w.-]+)', user_input, re.IGNORECASE)
                    fname_to_edit = match.group(2)
                    if open_file_for_editing(fname_to_edit):
                        spoken_response = f"Opened '{fname_to_edit}'."
                    else:
                        spoken_response = f"Sorry, couldn't open '{fname_to_edit}'."
                    response = spoken_response
                # --- AI Edit File (LLM-powered) ---
                elif re.match(r'ai edit file\s+([\w.-]+)\s+with instruction\s+(.*)', user_input, re.IGNORECASE | re.DOTALL):
                    match = re.match(r'ai edit file\s+([\w.-]+)\s+with instruction\s+(.*)', user_input, re.IGNORECASE | re.DOTALL)
                    filename_to_edit = match.group(1)
                    edit_instruction = match.group(2).strip()
                    logging.info(f"Starting AI edit for '{filename_to_edit}'. Instruction: '{edit_instruction}'")
                    original_code, read_error = read_full_file(filename_to_edit)
                    if read_error:
                        response = spoken_response = read_error
                    else:
                        ai_edit_prompt = f"""You are an expert programmer AI. Below is the content of the file '{filename_to_edit}' and a user request to modify it.\nUser Request: '{edit_instruction}'\n\nCurrent File Content:\n``` {original_code} ```\n\nYour task is to apply the user's request to the code. Return ONLY the complete, modified code for the entire file, enclosed in a single markdown code block (```). Do not add explanations before or after the code block. Ensure the modified code is syntactically correct and functional based on the request. If the request cannot be fulfilled reasonably, return the original code block unchanged.\n"""
                        logging.debug(f"Sending prompt to LLM for AI Edit:\n{ai_edit_prompt}")
                        try:
                            logging.info("Sending AI edit request to LLM...")
                            ai_response = chain.run(conversation_history="", user_input=ai_edit_prompt)
                            logging.debug(f"LLM raw response for AI edit:\n{ai_response}")
                            proposed_code = extract_code_from_response(ai_response)
                            if proposed_code:
                                if proposed_code.strip() == original_code.strip():
                                    response = "The AI didn't propose any changes based on your instruction."
                                    spoken_response = "The AI didn't propose any changes."
                                else:
                                    try:
                                        with open(filename_to_edit, 'w', encoding='utf-8') as f:
                                            f.write(proposed_code)
                                        response = f"AI edit complete! Changes applied to '{filename_to_edit}'."
                                        spoken_response = f"AI edit complete. Changes saved to {filename_to_edit}."
                                    except Exception as e:
                                        response = f"Sorry, failed to save changes to '{filename_to_edit}': {e}"
                                        spoken_response = response
                            else:
                                response = "Sorry, I couldn't extract the modified code from the AI response. No changes made."
                                spoken_response = "Sorry, I had trouble generating the changes."
                        except Exception as llm_err:
                            logging.error(f"Error during LLM call for AI edit: {llm_err}", exc_info=True)
                            response = "Sorry, an error occurred while communicating with the AI for editing."
                            spoken_response = response
                # --- Default: General Conversation ---
                else:
                    llm_response = add_hesitation() + chain.run(conversation_history=" ".join(conversation_history[-10:]), user_input=user_input)
                    response = llm_response
                    spoken_response = filter_text_for_speech(llm_response)
                # --- Process and Speak Response (Common for all commands) ---
                if response:
                    logging.info(f"Sam: {response}")
                    print(f"Sam: {response}")
                    if spoken_response:
                        speak_text(engine, spoken_response, is_speaking)
                    else:
                        logging.warning("Filtered response for speech is empty.")
                    conversation_history.append(f"User: {user_input}")
                    conversation_history.append(f"Sam: {response}")
                    conversation_history = conversation_history[-20:]
                # If exit command was processed, exit loop now
                if not running:
                    time.sleep(2)  # Allow TTS to finish
                    continue
        except queue.Empty:
            time.sleep(0.05)
            continue
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Shutting down.")
            running = False
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
            try:
                speak_text(engine, "Oops, an error occurred.", is_speaking)
            except Exception:
                pass
            time.sleep(1)
    # --- Cleanup (Voice Mode) ---
    logging.info("Shutting down voice chat...")
    if listener_stop_func:
        logging.info("Stopping background listener...")
        listener_stop_func(wait_for_stop=False)
    if tts_thread:
        logging.info("Signaling TTS worker to stop...")
        tts_queue.put(None)
        tts_queue.join()
        tts_thread.join(timeout=5)
        if tts_thread.is_alive():
            logging.warning("TTS thread did not exit cleanly.")
    if engine:
        try:
            with engine_lock:
                if hasattr(engine, '_inLoop') and engine._inLoop:
                    engine.endLoop()
        except Exception as e:
            logging.warning(f"Final TTS cleanup error: {e}")
    logging.info("Sam AI Assistant (Voice Mode) shut down.")
    print("\nSam AI Assistant (Voice Mode) shut down.")

# Main function - Mode Selection Logic
def main():
    """Main function to set up and run the assistant in voice or text mode."""
    config = load_config()
    llm = initialize_llm(config)
    if not llm:
        return  # Exit if LLM fails
    chat_chain = create_chat_chain(llm)
    tts_engine = initialize_tts(config)  # Initialize TTS (can be None)
    # Determine mode based on command-line argument
    text_mode = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--text-mode":
        text_mode = True
        print("Starting Sam in text-based testing mode (via argument)...")
        logging.info("Starting Sam in text-based testing mode...")
    else:
        print("Starting Sam in default voice recognition mode...")
        logging.info("Starting Sam in default voice recognition mode...")
    if text_mode:
        # Run Text-based mode
        text_based_chat(chat_chain, tts_engine, config)
    else:
        # Run Full speech recognition mode
        recognizer = initialize_recognizer(config)  # Initialize recognizer only for voice mode
        start_chat(chat_chain, tts_engine, recognizer, config)  # Start the voice chat loop

if __name__ == "__main__":
    main()