import streamlit as st
import MeCab
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from googletrans import Translator
from transformers import pipeline
import nltk
import random
# Initialize a transformer-based language generation pipeline
generator = pipeline('text-generation', model='gpt2')

def tokenize_japanese_text(text):
    mecab = MeCab.Tagger("-Owakati")
    tokens = mecab.parse(text).split()
    return tokens

def pos_tagging(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags

def named_entity_recognition(text):
    pos_tags = pos_tagging(text)
    tree = ne_chunk(pos_tags)
    named_entities = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NE':
            entity = ""
            for leaf in subtree.leaves():
                entity = entity + leaf[0] + " "
            named_entities.append(entity.strip())
    return named_entities

def translate_text(text):
    translator = Translator()
    translated = translator.translate(text, src='ja', dest='en')
    return translated.text

import random

def generate_quiz(user_input):
    # Tokenize the user input
    tokens = word_tokenize(user_input)

    # Randomly select a word to be the answer
    answer = random.choice(tokens)

    # Replace the answer with a blank in the question
    question = user_input.replace(answer, "______")

    return question, answer


st.title('Jisho Assistant')

# Create a list to store the chat history
chat_log = []
chat_placeholder = st.empty()
user_input = st.text_input("Enter a Japanese sentence:")

# Add a button for generating the quiz
if st.button('Generate Quiz'):
    if user_input:
        tokens = tokenize_japanese_text(user_input)
        chat_log.append(f"YOU: {user_input}")
        chat_log.append(f"Bot: Tokens: {tokens}")

        translation = translate_text(user_input)
        chat_log.append(f"Bot: Translation: {translation}")

        pos_tags = pos_tagging(translation)
        chat_log.append(f"Bot: Part-of-Speech Tags: {pos_tags}")

        named_entities = named_entity_recognition(translation)
        chat_log.append(f"Bot: Named Entities: {named_entities}")

        question, answer = generate_quiz(user_input)
        chat_log.append(f"Bot: Quiz: {question}")

        user_answer = st.text_input("Your answer:")
        if user_answer:
            if user_answer == answer:
                chat_log.append(f"User: {user_answer}")
                chat_log.append("Bot: Correct!")
            else:
                chat_log.append(f"User: {user_answer}")
                chat_log.append(f"Bot: Sorry, the correct answer is: {answer}")

# Update the chat log outside the if condition
chat_placeholder.write("Chat Log:")
for message in chat_log:
    chat_placeholder.write(message)