
import pyttsx3
# pyttsx3 enables the Spear, to communicate with the user audibly by 
# converting text into speech.

import datetime

import speech_recognition as sr
# speech_recognition enables the assistant to understand and process 
# voice commands from the user.

import wikipedia
# wikipedia library enhances the assistant's functionality by allowing it to 
# provide informative responses sourced from Wikipedia articles.

import nltk
# nltk is a function preprocesses text by tokenizing it, converting words to lowercase, 
# and removing stopwords. This preprocessing step is essential for preparing text data for 
# tasks like machine learning-based text classification.

import webbrowser
# The webbrowser module in Python provides a high-level interface for displaying 
# web-based documents to users

import os
# the os module enables the assistant to perform operating system-level operations such as
# file management and execution of external programs, enhancing its capabilities beyond simple 
# text or voice interactions.

import pandas as pd
# pandas is integral to handling and preprocessing the dataset, making it suitable for training
# a machine learning model to answer questions posed to the assistant.

from sklearn.feature_extraction.text import TfidfVectorizer
#  the TfidfVectorizer plays a crucial role in converting text data into numerical features
# suitable for training a machine learning model, allowing the assistant to answer questions
# based on the trained model.

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
# The RandomForestClassifier is instantiated with default parameters. RandomForest is a type of
# ensemble learning method that operates by constructing multiple decision trees during
# training and outputting the class that is the mode of the classification or
# mean prediction of the individual trees

import re

# Download NLTK data if not already downloaded
nltk.download('punkt')
# This helps computers break down text into smaller, more manageable parts, making it
# easier to analyze and understand.

nltk.download('stopwords')

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
# the sapi5 used  to convert Text into Speech.

voices = engine.getProperty('voices')
# getProperty('voices') is used to retrieve a list of available voices for text-to-speech conversion

engine.setProperty('voice', voices[0].id)

# Function to speak out the given audio
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
# runAndWait() ensures that text is spoken synchronously and that the program waits for
# the speech to finish before proceeding, enhancing the user experience

# Function to greet the user
def wish_me():
    hour = datetime.datetime.now().hour
    if 0 <= hour < 12:
        speak("Good morning master!")
    elif 12 <= hour < 18:
        speak("Good afternoon master!")
    else:
        speak("Good evening master!")
    speak("I am Spear! How can I assist you today?")

# Function to recognize voice command
def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("You can speak now. Spear is listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print(f'User said: {query}\n')

        
        dataset_path = 'C:\\Users\\hp\\Documents\\general_dataset\\abusive_words.csv'
        dataset = pd.read_csv(dataset_path)
        dataset_words = set(dataset['abuses'])
        
        # Check if any word from the dataset is present in the user's command
        user_words = set(query.lower().split())
        if dataset_words.intersection(user_words):
            speak("You should speak politely , it's bad for your character development !")
        
    except Exception as e:
        speak("will you Please kindly repeat...")
        return "None"
    return query

# Function to handle question answering using Wikipedia
def answer_question_wikipedia(query):
    try:
        speak('SEARCHING WIKIPEDIA')
        query = query.replace("wikipedia","")
        results = wikipedia.summary(query,sentences=2)
        speak("According to wikipedia")
        speak(results)
    except Exception as e:
        speak("I'm sorry, I couldn't find information on that topic.")


# cleaning text
def clean_text(text):
    # Remove special characters , puctuations and digits
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()
    # Remove extra whitespaces
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


# Function to preprocess text
def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = nltk.word_tokenize(cleaned_text)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(tokens)

# It splits the given text into individual words or tokens
# It converts all the words to lowercase letters
# It gets rid of common words
# It combines the cleaned-up words into a single string



# Function to handle question answering using the dataset
def answer_question(query, model, vectorizer):
    try:
        query_vectorized = vectorizer.transform([preprocess_text(query)])
        predicted_answer = model.predict(query_vectorized)[0]
        if predicted_answer:
            speak(predicted_answer)
        else:
            speak("I'm sorry, I couldn't find an answer in the dataset for that question.")
            print("Unable to find an answer in the dataset for the query:", query)
    except Exception as e:
        speak("I'm sorry, I don't have an answer to that question.")
        print("Error:", e)

# Function to train machine learning model using a dataset
def train_model(dataset_path):
    dataset = pd.read_csv(dataset_path)
    X = dataset['question']
    y = dataset['answer']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # random_state=42 ensures that the random processes involved in the code, such as shuffling
    #data or initializing model parameters, will give the same results every time you run the code
    

    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    X_train_vectorized = vectorizer.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer


# Function to take song name input from the user
def take_song_input():
    speak("Please tell me the name of the song you want to play.")
    song_name = take_command()
    return song_name


# Function to play a specific song in the directory
def play_song(music_dir, song_name):
    songs = os.listdir(music_dir)
    found = False
    for song in songs:
        if song_name.lower() in song.lower() and song.endswith('.mp3'):
            os.startfile(os.path.join(music_dir, song))
            found = True
            break
    
    if not found:
        speak(f"Sorry, I couldn't find the song {song_name}.")


# Function to open web browser web pages
def open_web_page(url):
    webbrowser.open(url)



# Main function
def main():
    wish_me()
    
    dataset_path = 'C:\\Users\\hp\\Documents\\general_dataset\\final_dataset.csv'
    model, vectorizer = train_model(dataset_path)

    while True:
        query = take_command().lower()
        if 'exit' in query:
            speak("Goodbye!")
            break
        elif 'search' in query:
            answer_question_wikipedia(query)
        elif 'play music' in query:
            music_dir = "D:\\music\\songs"
            song_name = take_song_input()
            play_song(music_dir, song_name)
        elif 'open youtube' in query:
            open_web_page("https://www.youtube.com/")
        elif 'open google' in query:
            open_web_page("https://www.google.com/")
        else:
            answer_question(query, model, vectorizer)




if __name__ == "__main__":
    main()
