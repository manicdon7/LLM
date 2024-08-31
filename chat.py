import pyttsx3
import speech_recognition as sr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
import time

# Initialize the LLM (Language Model)
def initialize_llm():
    llm: LLM = G4FLLM(
        model=models.gpt_4o,
        provider=Provider.Chatgpt4o, 
    )
    return llm

# Define the prompt template
def create_prompt_template():
    template = (
        "You are a personal assistant, your owner's name is Mani, "
        "and you will be loyal to your owner. The user says: {user_input}\n\nYour response:"
    )
    prompt = PromptTemplate(input_variables=["user_input"], template=template)
    return prompt

# Define the chat chain
def create_chat_chain(llm):
    prompt = create_prompt_template()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Initialize text-to-speech engine
def initialize_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set speaking speed
    # Select a female voice if available
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    else:
        engine.setProperty('voice', voices[0].id)
    return engine

# Initialize speech recognizer
def initialize_recognizer():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Adjust based on ambient noise
    recognizer.pause_threshold = 0.8    # Adjust based on speaking speed
    return recognizer

# Function to recognize speech
def recognize_speech(recognizer):
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            print("Sorry, there was an error with the speech recognition service.")
            return ""

# Function to speak text
def speak_text(engine, text):
    engine.say(text)
    engine.runAndWait()

# Function to start the chat
def start_chat(chain, engine, recognizer):
    print("Chatbot is ready! Say 'exit' to stop the conversation.\n")
    speak_text(engine, "Hello! I am your personal assistant. How can I help you today?")
    
    while True:
        user_input = recognize_speech(recognizer)
        
        if not user_input:
            continue  # Skip if no input was recognized
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            speak_text(engine, "Goodbye!")
            break
        
        # Get the chatbot response
        response = chain.run(user_input=user_input)
        print(f"Bot: {response}\n")
        
        # Speak the response
        speak_text(engine, response)
        
        # Small pause to ensure the TTS engine finishes before listening again
        time.sleep(1)

# Main function to run the chat application
def main():
    llm = initialize_llm()
    chat_chain = create_chat_chain(llm)
    tts_engine = initialize_tts()
    recognizer = initialize_recognizer()
    start_chat(chat_chain, tts_engine, recognizer)

if __name__ == "__main__":
    main()
