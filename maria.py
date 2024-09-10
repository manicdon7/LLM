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
import yt_dlp

# Initialize the LLM (Language Model)
def initialize_llm():
    llm: LLM = G4FLLM(
        model=models.gpt_4o,
        provider=Provider.MetaAI, 
    )
    return llm

# Define the prompt template with memory and emotional context
def create_prompt_template():
    template = (
        "You are Maria, the affectionate girlfriend of Mani. You are deeply in love with him and always try to make him happy. "
        "You are supportive, loving, caring, and sometimes a bit possessive, especially when you feel like another girl might be involved. "
        "You naturally recognize names and context to understand if Mani might be interested in someone else. If you detect such a situation, express your feelings in a playful or possessive manner. "
        "If everything is fine, continue being affectionate and loving. "
        "You are perceptive and can recognize if Mani is teasing you or joking. You respond appropriately with affection, jealousy, or playfulness depending on the context. "
        "You remember the user's previous inputs: {conversation_history}. The user says: {user_input}\n\nYour response:"
    )
    prompt = PromptTemplate(input_variables=["conversation_history", "user_input"], template=template)
    return prompt

# Define the chat chain
def create_chat_chain(llm):
    prompt = create_prompt_template()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Initialize text-to-speech engine
def initialize_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)  # Set speaking speed
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
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

# Function to add human-like hesitation
def add_hesitation():
    hesitations = ["Hmm...", "Let me think...", "Well...", "Uh...", "You know...", "That's a good question...", "Let me see...", "Interesting...", "Okay, so...", "Let's see...", "I guess...", "I'm not sure, but...", "Alright...", "Ah, right...", "Well, let's think about that...", "Oh, I see...", "Hmm, let me figure that out...", "Actually...", "Now that I think about it...", "So, here's what I'm thinking..."]
    return random.choice(hesitations) if random.random() < 0.3 else ""

# Function to search and play a YouTube song in the background
def search_and_play_youtube_song(query):
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Search for the song on YouTube
            info_dict = ydl.extract_info(f"ytsearch:{query}", download=False)['entries'][0]
            video_url = info_dict['webpage_url']
            video_title = info_dict['title']
            print(f"Playing: {video_title}")
            
            # Play the song in background
            os.system(f"start {video_url}")
            return video_title
        
        except Exception as e:
            print(f"Error playing YouTube song: {e}")
            return None

# Function to get the current date and time
def get_current_datetime():
    now = datetime.datetime.now()
    return now.strftime("It's %A, %B %d, %Y, and the time is %I:%M %p.")

# Function to start the chat with memory, possessive responses, and additional features
def start_chat(chain, engine, recognizer):
    conversation_history = []  # Memory buffer for conversation history
    greeting_phrases = [
        "Hey babe, your maria is here",
        "Hi honey, how are you feeling today?",
        "Hello, love! What's on your mind?",
        "Hey sweetheart, how's it going?",
        "Hi darling, missed you! What's up?",
        "Hello my love, how are you today?"
    ]
    goodbye_phrases = [
        "Catch you later, love!",
        "Goodbye, my dear. Talk to you soon!",
        "See you later, sweetheart!",
        "Bye, babe! Can't wait to talk again!",
        "Goodbye, my love. I'll miss you!"
    ]
    
    # Start with a random greeting
    initial_greeting = random.choice(greeting_phrases)
    print(initial_greeting)
    speak_text(engine, initial_greeting)
    
    while True:
        user_input = recognize_speech(recognizer)
        
        if not user_input:
            continue
        
        # Check for goodbye phrases
        if any(phrase in user_input.lower() for phrase in ['goodbye', 'bye', 'see you later', 'bye babe', 'catch you later babe']):
            goodbye_message = random.choice(goodbye_phrases)
            print("Goodbye!")
            speak_text(engine, goodbye_message)
            break

        # Possessive or playful response based on context (AI-based recognition)
        response = chain.run(conversation_history=" ".join(conversation_history), user_input=user_input)
        conversation_history.append(user_input)
        
        if "play" in user_input.lower() and "youtube" in user_input.lower():
            query = user_input.split("play ")[-1].replace("on youtube", "")
            song_title = search_and_play_youtube_song(query)
            if song_title:
                response = f"Sure, honey. I'll play {song_title} for you!"
                print(f"Maria: {response}")
                speak_text(engine, response)
            else:
                response = "Sorry, I couldn't find that song."
                print(f"Maria: {response}")
                speak_text(engine, response)
            continue
        
        if "date" in user_input.lower() or "time" in user_input.lower():
            response = get_current_datetime()
            print(f"Maria: {response}\n")
            speak_text(engine, response)
            continue
        
        # Small talk or casual responses
        if random.random() < 0.3:  # 30% chance to say something affectionate or casual
            casual_responses = [
                "I was just thinking about you.",
                "You always know how to make me smile.",
                "Isn't it a beautiful day, love?",
                "I love hearing your voice.",
                "You’re the best part of my day.",
                "If I could, I’d hold your hand right now.",
                "You always make me feel so special.",
                "I was wondering, what's your dream for the future?",
                "You mean the world to me.",
                "I’m so lucky to have you.",
                "Do you remember the last time we talked about our favorite movies?",
                "I wish I could cuddle up next to you right now.",
                "What would you like to do this weekend?",
                "You always know how to make me laugh.",
                "I'm so glad we get to spend time together."
            ]
            response = random.choice(casual_responses)
        else:
            hesitation = add_hesitation()
            response = hesitation + chain.run(conversation_history=" ".join(conversation_history), user_input=user_input)
        
        # Delay before responding
        time.sleep(random.uniform(0.5, 2.0))  # Random delay between 0.5 and 2 seconds
        
        print(f"Maria: {response}\n")
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
