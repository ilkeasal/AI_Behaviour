## OPENAI_API_KEY ACCESS :
import os
from dotenv import load_dotenv
import openai
import time
import random
import streamlit as st
from google.cloud import storage
import uuid
from uuid_shortener import UUIDShortener
import logging
from io import StringIO
from google.oauth2 import service_account
import streamlit_survey as ss

from datetime import datetime

current_time = datetime.now()
adjusted_time = current_time.strftime("%d:%m:%y %H:%M")



survey = ss.StreamlitSurvey()

openai.api_key = st.secrets["OPENAI_KEY"]
GOOGLE_APPLICATION_CREDENTIALS = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
# load_dotenv(".env")
# openai.api_key = os.getenv("OPENAI_KEY")
# openai.api_key = os.environ["OPENAI_KEY"]

if not openai.api_key:
   raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = openai.OpenAI(api_key=openai.api_key)
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain  # Check why am I using LLMChain instead of normal LLM again.
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase

st.title(" Your Favorite Physical Health Assistant :rocket:")
credentials = service_account.Credentials.from_service_account_info(GOOGLE_APPLICATION_CREDENTIALS)
## Neo4j :

###########
NEO4J_USERNAME = "neo4j"
# NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_KEY"]
# NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
neo4j_url = "neo4j+s://e8df9493.databases.neo4j.io"
project_id = st.secrets["GOOGLE_PROJECT_ID"]
# project_id = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]["project_id"]


AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
with GraphDatabase.driver(neo4j_url, auth=AUTH) as driver:
   driver.verify_connectivity()
   print("Connection established.")

session = driver.session()

graph = Neo4jGraph(url=neo4j_url, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

def upload_to_bucket(bucket_name, log_data, user_id):
   client = storage.Client(credentials=credentials, project=project_id)
   bucket = client.bucket(bucket_name)
   blob_name = f"logs/user_{user_id}.log"
   blob = bucket.blob(blob_name)

   blob.upload_from_string(log_data, content_type="text/plain")


if "log_buffer" not in st.session_state:
   st.session_state.log_buffer = StringIO()

# adding new log entries
# st.session_state.log_buffer.write("This is a new log entry from the chatbot!\n")


# Set up Logger :
logger = logging.getLogger("local_test_logger")
logger.setLevel(logging.DEBUG)

# FileHandler to write logs to the log file:
stream_handler = logging.StreamHandler(st.session_state.log_buffer)
stream_handler.setLevel(logging.DEBUG)

# Defining a log format :
formatter = logging.Formatter("%(asctime)s -%(levelname)s -%(message)s")
stream_handler.setFormatter(formatter)

# Add handler to logger

if not logger.handlers:
   logger.addHandler(stream_handler)


# Google cloud storage setup :

if "uploaded_to_bucket" not in st.session_state:
   st.session_state.uploaded_to_bucket = False

st.session_state.uploaded_to_bucket = False


def variable_nodes_names():
   result1 = session.run("MATCH(n:VARIABLE) RETURN n.name")
   result1_value = (result1.values())
   variable_nodes_dict = {}
   for item in result1_value:
       for x in item:
           variable_nodes_dict.update({x: 0})
   return variable_nodes_dict  # this returns a list of the node names.
   # I should exclude the concepts that are already talked about from this list.


concept_definitions_dict = {
   "attitude": "refers to beliefs about how harmful, healthy, enjoyable, and boring physical activity is perceived.",
   "self-efficacy": "refers to people's belief in their ability to engage in physical activity to achieve their fitness goals, even in the face of challenges.",
   "exercise self identity": "refers to the extent to which an individual sees exercise as a fundamental part of who they are.",
   "action planning": "refers to the process of making detailed plans about when, where, and how often one engages in physical activity.",
   "coping strategies": "refers to an individual's ability to manage tempting situations and cues that could disrupt their physical activity routine.",
   "intentions": "refers to the amount of physical activity an individual intends to do.",
   "habit strength": "refers to the degree of automaticity and consistency in an individual's physical activity behaviour often developed through a history of repeated actions.",
   "social support non-family": " refers to support for physical activity from friends, acquaintances, or co-workers.",
   "social support family": "refers to support for physical activity from family.",
   "mood": "refers to temporary state of mind defined by feelings and dispositions.",
   "self monitoring": "refers to an individual's ability to regularly track and evaluate their physical activity levels.",
   "social norms": "refers to the cultural and environmental influences shaping how individuals perceive and engage in physical activity."
}
constructs = ["Motivation", "Opportunity", "Capability", "Physical Activity"]
variables = ["coping strategies", "self-efficacy", "intentions", "habit strength", "exercise self identity",
            "social support non-family",
            "social support family", "mood", "self monitoring", "action planning", "attitude", "social norms"]


# memory = ConversationBufferMemory(return_messages=True)


def stop_or_continue(a_dict):  # This is not necessary right now.
   """Checks how many concepts are discussed.
   Based on that I will create a stop and continue buttons."""
   total_num = 0
   for value in a_dict.values():
       if value > 0:
           total_num += 1

   return total_num


def generate_response(response):
   """ Stream the response """

   for word in response:
       yield word
       time.sleep(0.02)


def choose_lowest_concept(my_dict):
   """Chooses the concept that is talked about the least"""
   number = 10000
   for x in my_dict.keys():
       num = my_dict[x]
       if num <= number:
           number = num
           name_of_concept = x

   return name_of_concept


def adjust_value_property(llm_answer):
   constructs = ["Motivation", "Opportunity", "Capability", "Physical Activity"]
   variables = ["coping strategies", "self-efficacy", "intentions", "habit strength", "exercise self identity",
                "social support non-family",
                "social support family", "mood", "self monitoring", "action planning", "attitude",
                "social norms"]
   print(llm_answer)
   if "Low" in llm_answer:
       llm_answer = llm_answer.replace("Low", "")
       llm_answer = llm_answer.strip()  # to get rid of the white space because it is like " Action Planning"
       if llm_answer in constructs:
           query = "MATCH(X:CONSTRUCT {name: '%s'}) SET X.value='Low'" % llm_answer
           session.run(query)
           return llm_answer
       elif llm_answer in variables:
           query = "MATCH(X:VARIABLE {name: '%s'}) SET X.value='Low'" % llm_answer
           print("Low!")
           session.run(query)
           return llm_answer

   elif "High" in llm_answer:
       llm_answer = llm_answer.replace("High", "")
       llm_answer = llm_answer.strip()
       if llm_answer in constructs:
           query = "MATCH(X:CONSTRUCT {name: '%s'}) SET X.value='High'" % llm_answer
           session.run(query)
           return llm_answer
       elif llm_answer in variables:
           print("High!")
           query = "MATCH(X:VARIABLE {name: '%s'}) SET X.value='High'" % llm_answer
           session.run(query)
           return llm_answer

   else:
       return "No concept"


def retrieve_concepts(main_problem_node):
   """Returns an empty dict if there are no HAS_EFFECTS"""
   main_problem = main_problem_node
   new_dict = {}
   result = session.run(
       "MATCH(V:VARIABLE)-[:HAS_EFFECT]->(e:VARIABLE{name:'%s'}) RETURN (V.name)" % main_problem_node).values()

   result_flattened = sum(result, [])  # flattens the list

   for concept in result_flattened:
       result = session.run(
           "MATCH(V:VARIABLE)-[:HAS_EFFECT]->(e:VARIABLE{name:'%s'}) RETURN (V.name)" % concept).values()
       result = sum(result, [])
       new_dict[concept] = result

   return new_dict


def initialize_users_level_dict(unstable_concept_dict):
   users_level = {}
   for key, values in unstable_concept_dict.items():

       if key not in users_level:
           users_level[key] = {"status": "Unknown", "dependencies": {}}

       for value in values:
           if value not in users_level[key]["dependencies"]:
               users_level[key]["dependencies"][value] = {"status": "Unknown"}

   return users_level


def assess_concepts_prompt(concept_name, user_prompt, chat_history):
   """Assess concepts with openai + formatted prompt"""
   formatted_prompt = f"""Respond to the  {user_prompt} with a friendly and relevant acknowledgement. If the users' input includes a question or request for clarification, address it directly. Additionally, if the user's input raises areas that could benefit from further exploration, such as challenges, concerns, or unclear aspects, address these by asking a relevant and neutral follow-up question. Then naturally steer the conversation toward discussing aspects related to {concept_name} about physical activity. {concept_name} {concept_definitions_dict[concept_name]}. When steering the conversation, ask exactly one direct and neutral question aimed at understanding the user's current behaviour or thoughts about {concept_name}. This question must assess their current level on {concept_name}, without making any assumptions about what the user does/thinks or does not do/think. Do not make assumptions about the user based solely on their input. Ensure that your response transitions smoothly from {user_prompt} and is open-ended to encourage discussion. Do not include any explanations, reflections, or commentary about the purpose, structure, or intent of your response, either implicitly or explicitly. Use relevant aspects from the chat history ({chat_history}) to maintain context but do not repeat previous responses verbatim. Each response should contribute new insights or questions to the conversation without breaking the flow or including unnecessary meta-information."""
   print(f" Assess concepts prompt : {assess_concepts_prompt}")
   completion = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{
           "role": "assistant",
           "content": formatted_prompt
       }]
   )
   st.session_state.log_buffer.write(f"ASSESS CONCEPTS PROMPT : {formatted_prompt}\n")
   response = completion.choices[0].message.content
   with st.chat_message("assistant"):
       response = st.write_stream(generate_response(response))
       st.session_state.log_buffer.write(f"ASSISTANT SAID : {response}\n")
       st.session_state.log_buffer.write("\n")
       st.session_state.messages.append({"role": "assistant", "content": response})


def give_advice_prompt(main_problem, user_prompt, chat_history):
   """Give advice with openai + formatted prompt"""
   st.session_state.advice_given = True
   formatted_prompt = f"""Start by acknowledging the user's input, {user_prompt} in a friendly and empathetic way.Reflect briefly on their perspective to show understanding. If the user's input includes a question or request for clarification, address it directly. The user will benefit from strategies related to {main_problem}. {main_problem} {concept_definitions_dict[main_problem]}. Based on the user's input,and the chat_history ({chat_history}) offer advice that to help the user with their {main_problem}.In your response, make assumptions or inferences only if they are clearly supported by the chat history. Do not include any explanations, reflections, or commentary about the purpose, structure, or intent of your response, either implicitly or explicitly.  Use relevant parts of the chat history, for context, but avoid repeating advice word-for-word. Make sure your response builds on the past conversations and adds new insights.If you made promises to the user such as providing specific recommendations in the previous conversation build on them in a consistent and supportive way. Phrase your recommendations in a unique, and varied language to keep the conversation fresh and engaging. Relate your advice directly to the user's input to show you're listening.Keep your responses concise, ideally under 250 tokens, while maintaining a complete thought."""
   st.session_state.log_buffer.write(f"GIVE ADVICE PROMPT : {formatted_prompt}\n")
   completion = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{
           "role": "assistant",
           "content": formatted_prompt
       }]
   )
   response = completion.choices[0].message.content
   with st.chat_message("assistant"):
       response = st.write_stream(generate_response(response))
       st.session_state.log_buffer.write(f"ASSISTANT ADVICE : {response}\n")
       st.session_state.log_buffer.write("\n")
       st.session_state.messages.append({"role": "assistant", "content": response})


def is_it_key(concept_name, unstable_concept_dict):
   """Here the unstable concept dict will be st.session_state.unstable_concept_dictionary"""
   if concept_name in unstable_concept_dict:
       st.session_state.its_key = True
   else:
       st.session_state.its_key = False


def filter_low_dependencies_as_list(user_level_dict):
   """This creates a dict with concept names with low status,
       including the concepts with low status that has effect on these concepts.
       If there is no 'Low' value it returns an empty dictionary.
       If the concept names status is Unknown but there are low dependencies it returns
       the concept name and the low dependencies.
       Example output : {'habit strength': ['mood', 'coping strategies', 'action planning'], 'coping strategies': [], 'exercise self identity': ['self-efficacy']}. """
   result = {}

   for key, details in user_level_dict.items():
       status = details["status"]  # here details is the value
       dependencies = details["dependencies"]

       # Collect dependencies with 'Low' status

       low_dependencies = [dep_key for dep_key, dep_details in dependencies.items() if dep_details["status"] == "Low"]

       if status == "Low" and low_dependencies:
           result[key] = low_dependencies

       elif status == "Low" and not low_dependencies:
           result[key] = []

       elif status == "Unknown" and low_dependencies:
           result[key] = low_dependencies

   return result


def give_advice_sent(users_level_dict):
   """This gets in the filtered users_level_dict and it returns a sentence that says the user struggles with this and that
   and the definitions.
   However it does not say give advice in this sentence.
   """
   sentence1 = "The user struggles with the following aspects related to physical activity: "

   for key, value in filtered_dict.items():
       sentence1 += f"{key}\n"
       sentence1 += f" - Definition: {key} {concept_definitions_dict[key]}\n"

       # Add influencing factors if present

       if value:  # if dependencies exist
           sentence1 += " - Influencing Factors: \n"

           for item in value:
               sentence1 += f"{item}: {concept_definitions_dict[item]}\n"

       else:

           sentence1 += f" - No influencing factors identified. \n"

       sentence1 += "\n"

   return sentence1


def give_advice_users_level(give_advice_sentences):
   """Give advice sentence with openai + formatted prompt"""
   st.session_state.advice_given = True
   combined_input = "In your response, make assumptions or inferences only if they are clearly supported by the chat history. Do not include any explanations, reflections, or commentary about the purpose, structure, or intent of your response, either implicitly or explicitly."
   formatted_prompt = f"""Start by acknowledging the user's input, {user_prompt} in a friendly and empathetic way. Reflect briefly on their perspective to show understanding. If the user's input includes a question or request for clarification, address it directly. {give_advice_sentences}.Based on this information, user's input : {user_prompt}, and chat history: {chat_history} offer advice to help the user become more physically active.{combined_input}. Use relevant parts of the chat history, for context, but avoid repeating advice word-for-word. Make sure your response builds on the past conversations and adds new insights.If you made promises to the user such as providing specific recommendations in the previous conversation build on them in a consistent and supportive way. Phrase your recommendations in a unique, and varied language to keep the conversation fresh and engaging. Relate your advice directly to the user's input to show you're listening. Keep your responses concise, ideally under 250 tokens, while maintaining a complete thought."""
   st.session_state.log_buffer.write(f"GIVE ADVICE USERS LEVEL PROMPT : {formatted_prompt}\n")
   completion = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[
           {
               "role": "assistant",
               "content": formatted_prompt
           }
       ]
   )
   response = completion.choices[0].message.content
   with st.chat_message("assistant"):
       response = st.write_stream(generate_response(response))
       st.session_state.log_buffer.write(f"ASSISTANT ADVICE : {response}\n")
       st.session_state.log_buffer.write("\n")
       st.session_state.messages.append({"role": "assistant", "content": response})


def clarification_question(last_asked_concept, chat_history):
   assistant_answer = " ".join(chat_history[-2])
   user_answer = " ".join(chat_history[-1])
   formatted_prompt = f"""The user's response '{user_answer}' did not appropriately answer the question '{assistant_answer}'. Acknowledge the user's response naturally and empathetically. If the users' input includes a question or request for clarification, address it directly. Then seamlessly transition to rephrasing and simplifying the original question to steer the conversation back to the concept '{last_asked_concept}'. {last_asked_concept} {concept_definitions_dict[last_asked_concept]}. Ensure that you only ask one revised question and that question is direct and neutral and aims to understand user's current behaviour or thoughts about {last_asked_concept} while flowing naturally from the acknowledgement. Your output should be concise, with a smooth transition and a revised question. Take the chat history into account {chat_history} so that you do not repeat yourself. It is very important that you do not ask the same question word-by-word. Do not include any explanations, reflections, or commentary about the purpose or structure of your response."""
   st.session_state.log_buffer.write(f"CLARIFICATION QUESTION PROMPT : {formatted_prompt}\n")
   completion = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[
           {
               "role": "assistant",
               "content": formatted_prompt
           }
       ]
   )
   response = completion.choices[0].message.content
   with st.chat_message("assistant"):
       response = st.write_stream(generate_response(response))
       st.session_state.log_buffer.write(f"ASSISTANT SAID: {response}\n")
       st.session_state.log_buffer.write("\n")
       st.session_state.messages.append({"role": "assistant", "content": response})


def validation_question(last_asked_concept, chat_history):
   """Currently returns 'yes' or 'no' and a short explanation. Later on get rid of the explanation"""
   assistant_answer = " ".join(chat_history[-2])
   user_answer = " ".join(chat_history[-1])
   validation_question = (
       f"Does the user's response '{user_answer}' appropriately answer the question asked by the assistant '{assistant_answer}' to assess the users' level on {last_asked_concept}? {last_asked_concept} {concept_definitions_dict[last_asked_concept]}. Assess whether the user's response provides sufficient information-explicitly or implicitly-about their behaviours, attitudes, or practices related to {last_asked_concept}. Explicit information includes direct affirmations or denials (e.g., 'Yes, I do.', or 'No I do not'), while implicit information may involve reasoning, beliefs, or examples that clearly suggest a level of engagement (either low or high) with {last_asked_concept}. Responses that describe behaviours or beliefs indirectly connected to {last_asked_concept} can also be considered sufficient if they imply the user's level. For example, a response that describes consistent or inconsistent behaviours, attitudes, or practices directly tied to {last_asked_concept} can be enough to make an assessment. If the user's response provides enough information to evaluate their level on {last_asked_concept}, return 'yes' and provide a brief explanation of why it is sufficient. If the user's response does not address the concept or is too vague to make an assessment, return 'no' and provide a brief explanation of what is missing.")
   st.session_state.log_buffer.write(f"VALIDATION QUESTION PROMPT : {validation_question}\n")
   completion = client.chat.completions.create(model="gpt-4o-mini", messages=[
       {"role": "assistant",
        "content": validation_question}
   ])
   evaluation = completion.choices[0].message.content
   response = evaluation.lower()
   st.session_state.log_buffer.write(
       f"The answer of the GPT4o-mini in response to the validation question : {response}\n")
   st.session_state.log_buffer.write("\n")
   return response


def level_check_after_validation(user_answer, last_asked_concept):
   """Returns Low/ High of last asked concept/validated concept."""
   completion = client.chat.completions.create(model="gpt-4o-mini",
                                               messages=[
                                                   {"role": "assistant",
                                                    "content": f"""Based on the user's input : {user_answer}, decide if the level of {last_asked_concept} about physical activity is 'Low' or 'High'.
                                                                               A 'High' level indicates that the user demonstrates a strong, consistent, or positive engagement with the concept {last_asked_concept}. {last_asked_concept} {concept_definitions_dict[last_asked_concept]},
                                                                               A 'Low' level indicates minimal, inconsistent, or negative engagement with the concept.
                                                                              ONLY return 'Low {last_asked_concept}' or 'High {last_asked_concept}', with NO explanation or extra details."""}
                                               ]
                                               )
   level = (completion.choices[
                0].message.content).lower()  # should return either "Low {last_asked_concept} or High {last_asked_concept}
   st.session_state.log_buffer.write(f"After the validation, the decided level of the user is {level}\n")
   st.session_state.log_buffer.write("\n")
   return level


def stop_button():
   st.balloons()
   st.session_state.log_buffer.write("STOP BUTTON PRESSED!\n")
   st.session_state.start_experiment = "post-survey"


def stop_or_continue(a_dict):
   """Checks how many concepts are discussed.
   Based on that I will use the stop button."""
   total_num = 0
   for value in a_dict.values():
       if value > 0:
           total_num += 1

   return total_num

def submit_submit_function():
   st.session_state.start_experiment = "experiment"
   st.session_state.log_buffer.write("PHY ASSESSMENT!\n")
   st.session_state.log_buffer.write(f"Days input : {days_input}\n")
   st.session_state.log_buffer.write(f"Minutes input : {hours_input}\n")
   st.session_state.log_buffer.write(f"Strength input : {strength_input}\n")
   total_activity_levels = days_input * hours_input
   st.session_state.log_buffer.write(f"Total activity : {total_activity_levels}\n")
   st.session_state.log_buffer.write(f"Chatbot user experience : {chatbot_use} \n")
   st.session_state.log_buffer.write(f"User age : {age}\n")
   st.session_state.log_buffer.write("\n")
   if total_activity_levels >=150 and strength_input>=2:
       st.session_state.log_buffer.write("ACTIVE\n")
   else:
       st.session_state.log_buffer.write("NOT ACTIVE\n")

   st.session_state.log_buffer.write(f"\n")

def post_survey_submit():
   st.session_state.log_buffer.write("Post Survey Results : \n")
   st.session_state.log_buffer.write(f"Question 1 answer : {question1}\n")
   st.session_state.log_buffer.write(f"Question 2 answer : {question2}\n")
   st.session_state.log_buffer.write(f"Question 3 answer : {question3}\n")
   st.session_state.log_buffer.write(f"Question 4 answer : {question4}\n")
   st.session_state.log_buffer.write(adjusted_time)
   st.session_state.log_buffer.write("\n")
   st.session_state.start_experiment = "stop-experiment"
   st.session_state.save_conversation = True

def consent_submit():
   st.session_state.start_experiment = "pre-survey"


def high_level_praise_advice(user_high,concept_definitions):
   """ Here user_high is st.session_state.user_high and concept definitions is concept_definitions_dict"""
   high_level_sent = "The user demonstrates high levels of "
   concept_names = ", ".join(user_high)

   first_part = high_level_sent + concept_names + "."

   second_part = " "

   for concept in user_high:
      individual_sentence = concept + " " + concept_definitions[concept]

      second_part+= " " + individual_sentence

   full_sentence = first_part + second_part


   return full_sentence



def generic_advice_assess_concepts(high_level_praise_sent,concept_name,user_prompt,chat_history):
   """This function gets the output from the high_level_praise_advice function"""
   st.session_state.advice_given = True
   generic_advice_prompt = f"""Respond to the  {user_prompt} with a friendly and relevant acknowledgement. If the users' input includes a question or request for clarification, address it directly. {high_level_praise_sent}  Based on this, provide a clear and specific piece of advice that encourages the user to maintain their positive habits while offering tips to sustain long-term success. In your advice, make assumptions only if they are clearly supported by the chat history : {chat_history}. After providing the advice, naturally steer the conversation toward discussing aspects related to {concept_name} about physical activity. {concept_name} {concept_definitions_dict[concept_name]}.Ask exactly one direct and neutral question aimed at understanding the user's current behaviour or thoughts about {concept_name}.  The question must assess their current level on {concept_name} without making any assumptions about what the user does/thinks or does not do/think. Ensure that your response transitions smoothly from {user_prompt} and is open-ended to encourage discussion. Do not include any explanations, reflections, or commentary about the purpose, structure, or intent of your response, either implicitly or explicitly. Use relevant parts of the chat history but avoid repeating your responses. Make sure your responses build on the past conversation and adds new insights without breaking the flow or including unnecessary meta-information."""

   completion = client.chat.completions.create(model="gpt-4o-mini",messages=[{
   "role":"assistant",
   "content":generic_advice_prompt
   }])

   response = completion.choices[0].message.content

   st.session_state.log_buffer.write(f"GENERIC ADVICE PROMPT IS BEING USED! {generic_advice_prompt}\n")
   st.session_state.log_buffer.write(f"USER HIGH NEW : {st.session_state.user_high}\n")
   st.session_state.log_buffer.write("\n")
   

   with st.chat_message("assistant"):
      response = st.write_stream(generate_response(response))
      st.session_state.log_buffer.write(f"ASSISTANT SAID : {response}\n")
      st.session_state.log_buffer.write("\n")
      st.session_state.messages.append({"role": "assistant", "content": response})



## Streamlit Initialization

if "messages" not in st.session_state:
   st.session_state["messages"] = [{"role": "assistant",
                                    "content": "Hi!🤗 What's your experience with staying active? Is there anything you'd like to explore or talk about?"}]

if "unstable_concept_dict" not in st.session_state:
   st.session_state.unstable_concept_dict = {}

if "its_key" not in st.session_state:
   st.session_state.its_key = False

if "question_validation" not in st.session_state:
   st.session_state.question_validation = False

if "last_asked_concept" not in st.session_state:
   st.session_state.last_asked_concept = str()

if "turn_on" not in st.session_state:
   st.session_state.turn_on = False  # This one will turn True when we start working on unstable concept dict and False when we are done with it, when users level is complete

if "users_level" not in st.session_state:
   st.session_state.users_level = {}

if "all_concepts" not in st.session_state:
   st.session_state.all_concepts = variable_nodes_names()

if "main_problem_concept" not in st.session_state:
   st.session_state.main_problem_concept = str()

if "last_asked_key" not in st.session_state:
   st.session_state.last_asked_key = str()

if "value" not in st.session_state:
   st.session_state.value = str()

if "validation_repeat" not in st.session_state:
   st.session_state.validation_repeat = int()

if "active_conversation" not in st.session_state:
   st.session_state.active_conversation = True

if "asked_concepts" not in st.session_state:
   st.session_state.asked_concepts = []

if "start_experiment" not in st.session_state:
   st.session_state.start_experiment = "consent"

if "start_time" not in st.session_state:
   st.session_state.start_time = adjusted_time

if "advice_given" not in st.session_state:
   st.session_state.advice_given = False

if "user_high" not in st.session_state:
   st.session_state.user_high = []

if "save_conversation" not in st.session_state:
   st.session_state.save_conversation = False 

if "all_known_concepts" not in st.session_state:
   st.session_state.all_known_concepts = {"concepts":{}}

if "experiment_condition" not in st.session_state:
   st.session_state.experiment_condition = random.choice([1,2])
   if st.session_state.experiment_condition == 1:
      st.session_state.log_buffer.write(" 1 : EXPERIMENT CONDITION \n")
      st.session_state.log_buffer.write("\n")
   else:
      st.session_state.log_buffer.write(" 2 : CONTROL CONDITION \n")
      st.session_state.log_buffer.write("\n")
      

if st.session_state.start_experiment =="consent":
   st.text("""Welcome, and thank you for participating in this experiment. This experiment is part of a research project conducted by the AI & Behaviour group at the Vrije Universiteit Amsterdam. It involves interacting with a chatbot in a conversation focused on physical activity. It consists of three parts. In the first part, you will be asked a few questions about your age, prior experience with chatbots, and physical activity levels. This part will take about 2-3 minutes to complete.\n
As the main phase of this experiment, the second part involves a conversation with a chatbot, which is expected to last around 15 minutes. During the conversation, a ‘Stop the conversation’ button will appear at a certain point, giving you the option to end the conversation. Clicking this button will take you to the final part of the experiment, consisting of post-survey questions, which will take approximately 2-3 minutes to complete. If you wish to continue the conversation, no action is required-simply keep interacting with the chatbot. The ‘Stop the conversation’ button will become available at specific times, allowing you to stop the conversation later if you choose. If the button is not clicked, the conversation will continue indefinitely.\n
While interacting with the chatbot, you will notice the word ‘Running’ displayed in the top-right corner after you provide an input. This indicates that the chatbot is processing your input. The ‘Running’ indicator will disappear once the chatbot has completed its response. Please wait for the chatbot to finish generating its response before entering a new input. Similarly, during the first and final parts of the experiment, the ‘Running’ indicator will signify that your answer is being processed. Please wait for the ‘Running’ indicator to disappear before proceeding to the next question. After completing the final question and once the ‘Running’ indicator disappears, you may click the ‘Submit’ button.\n
Please be aware that while your conversation with the chatbot will be saved for research purposes, your privacy is our priority. The conversation will not be linked to your identity. Instead, you will be assigned a randomly generated ID, and all data will be associated with this ID only. We want to ensure that you feel comfortable throughout the process. You are free to stop participating in the experiment at any time by simply closing your browser, without needing to provide any reason. If you choose to do so, all data associated with your session will be removed. For participants who continue with the experiment until the end, the information collected will be used solely for the purpose of this study. 
   """)
   st.text("I have read and understood the information provided about the experiment and agree to participate voluntarily.")
   st.button("Submit",on_click=consent_submit)



if st.session_state.start_experiment=="pre-survey":
   #st.write("How often do you use chatbots?")
   st.session_state.log_buffer.write(adjusted_time)
   st.session_state.log_buffer.write("\n")
   age = st.number_input("What is your age?",min_value=18, max_value=64)
   options = {
       1:"Never",
       2:"1-2 times a year",
       3:"1-2 times a month",
       4:"1-2 times a week",
       5:"Daily"
   }
   chatbot_use= st.radio("How often do you use chatbots? (For example, AI-based chatbots like ChatGPT that can engage in a conversation with you.) ", options.keys(),index=None,format_func=lambda x:options[x])
   #st.session_state.log_buffer.write(f"Chatbot user experience : {response} \n")
   days_input = st.number_input("On average, how many days per week do you engage in moderate to strenuous exercise?", min_value=0,max_value=7)
   hours_input = st.number_input("On average, how many minutes do you engage in exercise at this level?",min_value=0,max_value=240)
   strength_input = st.number_input("On average, how many days a week do you perform muscle strengthening exercises, such as body weight exercises or resistance training?",min_value=0, max_value=7)
   st.markdown(f"""
       * Number of days: {days_input}
       * Number of minutes: {hours_input}
       * Strength activity: {strength_input}
       """)
   submit_button = st.button("Submit",on_click=submit_submit_function)


if st.session_state.experiment_condition == 1:

   st.write("EXPERIMENT CONDITION")

   if "user_id" not in st.session_state:
      st.session_state.user_id =("EXP_" + (UUIDShortener.encode(str(uuid.uuid4()))))

   if st.session_state.start_experiment == "experiment":
   
      for message in st.session_state["messages"]:
          with st.chat_message(message["role"]):
              st.markdown(message["content"])
   
   
      if user_prompt := st.chat_input("Want to share some thoughts?"):
          st.session_state.messages.append({"role": "user", "content": user_prompt})
          st.session_state.log_buffer.write(f"USER SAID : {user_prompt}\n")
          st.session_state.log_buffer.write("\n")
   
          with st.chat_message("user"):
              st.markdown(user_prompt)
   
          # GPTo4-Mini Start
   
          completion = client.chat.completions.create(
              model="gpt-4o-mini",
              messages=[
                  {"role": "assistant",
                   "content": f"""Among all the concepts mentioned in {variables} decide which one : {user_prompt} relates to? Only return the name of the concept. If the {user_prompt} does not directly relate to physical activity or explicitly address any of the listed concepts, includes general daily tasks like calling someone or waking up, consists of a question, or seeks clarification about a concept, return 'None of the above.' Only statements that clearly provide specific information about physical activity should be categorized. If the input explicitly asks for a definition, explanation, or meaning of a concept (e.g., 'What is coping strategies?','Can you explain action planning?') categorize it as 'None of the above'. When determining the concept take the {user_prompt} and chat history : {st.session_state.messages} into account. Check the chat history to determine if the user's input reflects a positive or neutral acknowledgement of a previously given recommendation, if so, categorize it as 'None of the above'. If the input : {user_prompt} points for any explicit or implicit problem or challenge related to physical activity, focus on categorizing it under the concept most closely related to that problem. Attitude refers to beliefs about how harmful, healthy, enjoyable, and boring physical activity is perceived. Self-efficacy refers to people's belief in their ability to engage in physical activity to achieve their fitness goals, even in the face of challenges. Exercise self identity refers to the extent to which an individual sees exercise as a fundamental part of who they are. Action planning refers to the process of making detailed plans about when, where, and how often one engages in physical activity. Coping strategies refers to an individual's ability to manage tempting situations and cues that could disrupt their physical activity routine. Intentions refers to the amount of physical activity an individual intends to do. Habit strength refers to the degree of automaticity and consistency in an individual's physical activity behaviour often developed through a history of repeated actions. Social support non-family refers to support for physical activity from friends, acquaintances, or co-workers. Social support family refers to support for physical activity from family. Mood refers to temporary state of mind defined by feelings and dispositions. Self monitoring refers to an individual's ability to regularly track and evaluate their physical activity levels. Social norms refers to the cultural and environmental influences shaping how individuals perceive and engage in physical activity."""}
              ]
          )
          construct_prompt = completion.choices[0].message.content
          construct_prompt = construct_prompt.lower()
   
          # stop_button = st.button("Stop the conversation.")
   
          if "none of the above" in construct_prompt:
              construct_name_level = "None of the above."
   
          else:
              try:
                  completion = client.chat.completions.create(
                      model="gpt-4o-mini",
                      messages=[
                          {
                              "role": "assistant",
                              "content": f"""Based on the user's input : {user_prompt}, decide if the level of {construct_prompt} about physical activity is 'Low' or 'High'.
                                       A 'High' level indicates that the user demonstrates a strong, consistent, or positive engagement with the concept {construct_prompt}. {construct_prompt} {concept_definitions_dict[construct_prompt]}.
                                       A 'Low' level indicates minimal, inconsistent, or negative engagement with the concept.
                                       ONLY return 'Low {construct_prompt}' or 'High {construct_prompt}', with NO explanation or extra details."""
                          }
                      ]
                  )
                  construct_name_level = completion.choices[0].message.content
   
                  print(construct_name_level)  # returns Low/High {construct_name}
   
              except KeyError:
                  print("Error handling taking place.")
                  completion = client.chat.completions.create(
                      model="gpt-4o-mini",
                      messages=[
                          {"role": "assistant",
                           "content": f"""Among all the concepts mentioned in {variables} decide which one : {user_prompt} relates to? Only return the name of the concept. If the {user_prompt} does not directly relate to physical activity or explicitly address any of the listed concepts, includes general daily tasks like calling someone or waking up, or consists of a question return 'None of the above.' Only statements that clearly provide specific information about physical activity should be categorized. When determining the concept take the {user_prompt} and chat history : {st.session_state.messages} into account. Check the chat history to determine if the user's input reflects a positive or neutral acknowledgement of a previously given recommendation, if so, categorize it as 'None of the above'. Attitude refers to beliefs about how harmful, healthy, enjoyable, and boring physical activity is perceived. Self-efficacy refers to people's belief in their ability to engage in physical activity to achieve their fitness goals, even in the face of challenges. Exercise self identity refers to the extent to which an individual sees exercise as a fundamental part of who they are. Action planning refers to the process of making detailed plans about when, where, and how often one engages in physical activity. Coping strategies refers to an individual's ability to manage tempting situations and cues that could disrupt their physical activity routine. Intentions refers to the amount of physical activity an individual intends to do. Habit strength refers to the degree of automaticity and consistency in an individual's physical activity behaviour often developed through a history of repeated actions. Social support non-family refers to support for physical activity from friends, acquaintances, or co-workers. Social support family refers to support for physical activity from family. Mood refers to temporary state of mind defined by feelings and dispositions. Self monitoring refers to an individual's ability to regularly track and evaluate their physical activity levels. Social norms refers to the cultural and environmental influences shaping how individuals perceive and engage in physical activity."""}
                      ]
                  )
                  construct_prompt = completion.choices[0].message.content
                  construct_prompt = construct_prompt.lower()
                  print(construct_prompt)
                  completion = client.chat.completions.create(
                      model="gpt-4o-mini",
                      messages=[
                          {
                              "role": "assistant",
                              "content": f"""Based on the user's input : {user_prompt}, decide if the level of {construct_prompt} about physical activity is 'Low' or 'High'.
                                                   A 'High' level indicates that the user demonstrates a strong, consistent, or positive engagement with the concept {construct_prompt}. {construct_prompt} {concept_definitions_dict[construct_prompt]}.
                                                   A 'Low' level indicates minimal, inconsistent, or negative engagement with the concept.
                                                   ONLY return 'Low {construct_prompt}' or 'High {construct_prompt}', with NO explanation or extra details."""
                          }
                      ]
                  )
                  construct_name_level = completion.choices[0].message.content
   
                  print(construct_name_level)  # returns Low/High {construct_name}
   
          concept_level = adjust_value_property(
              construct_name_level)  # this either returns the name of the concept, or no concept.
          print(f"This is the concept level : {concept_level}")
          chat_history = [(message["role"], message["content"]) for message in st.session_state.messages]
          print(f"This is chat history: {chat_history}")
   
          print(f"This is the construct prompt : {construct_prompt}")
          print(f"This is the construct name level : {construct_name_level}")
          st.session_state.log_buffer.write(f"LEVEL OF CONCEPT : {construct_name_level}\n")
   
          #if len(chat_history) > 25:
              #chat_history = chat_history[10:]
   
          if st.session_state.question_validation:
              if st.session_state.last_asked_concept == construct_prompt:  # We do not need to validate
                  st.session_state.all_concepts[st.session_state.last_asked_concept] += 1
                  print(f"We do not need to validate because {construct_prompt}=={st.session_state.last_asked_concept}")
                  print(f"All concepts:  {st.session_state.all_concepts}")
                  st.session_state.log_buffer.write(
                      f"We do not need to validate because {construct_prompt}=={st.session_state.last_asked_concept}\n")
                  st.session_state.log_buffer.write(f"ALL CONCEPTS :  {st.session_state.all_concepts}\n")
                  st.session_state.log_buffer.write("\n")
                  st.session_state.validation_repeat = 0
                  st.session_state.question_validation = False
                  # is_it_key(concept_name=st.session_state.last_asked_concept,unstable_concept_dict=st.session_state.unstable_concept_dict) #here we turn on the key.
   
                  if st.session_state.its_key:  # if what we are validating is a key
                      print("What we have just validated was a key.")
                      st.session_state.log_buffer.write("VALIDATED A KEY.\n")
                      st.session_state.log_buffer.write("\n")
   
                      if "Low" in construct_name_level:
                          st.session_state.users_level[st.session_state.last_asked_concept]["status"] = "Low"
                          print(st.session_state.users_level)
                          st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                          st.session_state.log_buffer.write("\n")
                          if st.session_state.unstable_concept_dict[
                              st.session_state.last_asked_concept]:  # if there are questions to ask about dependencies:
                              st.session_state.log_buffer.write(
                                  f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                              st.session_state.its_key = False
                              random_chosen_concept = random.choice(
                                  st.session_state.unstable_concept_dict[st.session_state.last_asked_concept])
                              st.session_state.value = random_chosen_concept  # maybe we don't even need this and we can just use last_asked_concept ?
                              st.session_state.last_asked_concept = random_chosen_concept
                              print(
                                  f"The value for the {st.session_state.last_asked_key} was low and it has dependencies. The chosen dependency to ask a question about is {st.session_state.last_asked_concept} ")
                              st.session_state.log_buffer.write(
                                  f"The value for the {st.session_state.last_asked_key} was low and it has dependencies. The chosen dependency to ask a question about is {st.session_state.last_asked_concept}\n")
                              st.session_state.log_buffer.write("\n")
                              assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                     user_prompt=user_prompt, chat_history=chat_history)
                              st.session_state.question_validation = True
   
   
                          else:  # If there are no questions to ask about dependencies because dependencies do not exist.
                              # del st.session_state.unstable_concept_dict[st.session_state.last_asked_concept]
                              del st.session_state.unstable_concept_dict[st.session_state.last_asked_key]
                              # Check if there is anything in the unstable concept dict, if not then give advice based on users level.
                              if st.session_state.unstable_concept_dict:
                                  st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                  st.session_state.last_asked_concept = st.session_state.last_asked_key
                                  st.session_state.log_buffer.write(
                                      f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                  # st.session_state.all_concepts[st.session_state.last_asked_concept] += 1
                                  st.session_state.its_key = True
                                  st.session_state.question_validation = True
                                  assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                         user_prompt=user_prompt, chat_history=chat_history)
   
                              else:
                                  st.session_state.question_validation = False
                                  filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                  sentence = give_advice_sent(filtered_dict)
                                  print(f"The sentence is :  {sentence}")
                                  st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                                  st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                                  st.session_state.log_buffer.write("\n")
                                  print(st.session_state.users_level)
                                  give_advice_users_level(sentence)
                                  st.session_state.unstable_concept_dict = {}
                                  st.session_state.users_level = {}
                                  if stop_or_continue(st.session_state.all_concepts) > 3:
                                      st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                      st.session_state.log_buffer.write("\n")
                                      stop_button = st.button("Stop the conversation.", on_click=stop_button)
                                  # I am currently not changing st.session_state.its_key to True but maybe I should. Look at it later.
   
                          # I need to keep asking qs about the dependencies if they exist.Else, give advice. If it is not key but value then it must belong to to last asked key that I name it as first key.
                          # Then if it is validated, delete it from unstable concept dict list with querying the last asked key = first key.
   
                      else:  # if "High" in construct name level
                          st.session_state.users_level[st.session_state.last_asked_concept]["status"] = "High"
                          st.session_state.user_high.append(st.session_state.last_asked_concept)
                          del st.session_state.unstable_concept_dict[st.session_state.last_asked_concept]
                          print(st.session_state.users_level)
                          st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                          st.session_state.log_buffer.write("\n")
                          # I need to praise then ask a question if there are still things to ask a question about. If not, I need to give advice.
                          if st.session_state.unstable_concept_dict:  # if there are still things to ask a question about.
                              st.session_state.log_buffer.write(
                                  f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                              st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                              st.session_state.last_asked_concept = st.session_state.last_asked_key
                              st.session_state.its_key = True
                              st.session_state.question_validation = True
                              assess_concepts_prompt(concept_name=st.session_state.last_asked_key, user_prompt=user_prompt,
                                                     chat_history=chat_history)
                              st.session_state.log_buffer.write("\n")
   
                          else:  # If there is nothing else to ask a question about, then give advice.
                              st.session_state.question_validation = False
                              filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                              sentence = give_advice_sent(filtered_dict)
                              print(f"The sentence is :  {sentence}")
                              st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                              st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}")
                              st.session_state.log_buffer.write("\n")
                              print(st.session_state.users_level)
                              give_advice_users_level(sentence)
                              st.session_state.unstable_concept_dict = {}
                              st.session_state.users_level = {}
                              if stop_or_continue(st.session_state.all_concepts) > 3:
                                  st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                  st.session_state.log_buffer.write("\n")
                                  stop_button = st.button("Stop the conversation.", on_click=stop_button)
                          # else: #if there is nothing to ask. It is time to give advice.
   
                  else:  # if we are validating a value
                      print("What we have just validated is a value")
                      st.session_state.log_buffer.write("VALIDATED A VALUE! \n")
                      if "Low" in construct_name_level:
                          st.session_state.users_level[st.session_state.last_asked_key]["dependencies"][
                              st.session_state.last_asked_concept]["status"] = "Low"
                          st.session_state.unstable_concept_dict[st.session_state.last_asked_key].remove(
                              st.session_state.last_asked_concept)
                          # st.session_state.all_concepts[st.session_state.last_asked_concept] += 1
                          print(st.session_state.users_level)
                          st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                          st.session_state.log_buffer.write("\n")
                          # check if there are still other dependencies
                          if st.session_state.unstable_concept_dict[
                              st.session_state.last_asked_key]:  # if there are other dependencies to ask a q about.
                              random_chosen_concept = random.choice(
                                  st.session_state.unstable_concept_dict[st.session_state.last_asked_key])
                              print(f"We are validating a value and the chosen value is : {random_chosen_concept}")
                              st.session_state.log_buffer.write(
                                  f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                              st.session_state.log_buffer.write(
                                  f"We are validating a value and the chosen value is : {random_chosen_concept}\n")
   
                              st.session_state.its_key = False
                              st.session_state.last_asked_concept = random_chosen_concept
                              st.session_state.question_validation = True
                              st.session_state.value = random_chosen_concept
                              assess_concepts_prompt(concept_name=random_chosen_concept, user_prompt=user_prompt,
                                                     chat_history=chat_history)
                              st.session_state.log_buffer.write("\n")
   
                          else:  # if there are no other dependencies to ask a q about
                              del st.session_state.unstable_concept_dict[st.session_state.last_asked_key]
   
                              if st.session_state.unstable_concept_dict:  # if there are other keys to ask a question about from the list.
                                  # This indicates that there are no other dependencies to ask a question about. Then I need to delete the key as well
                                  # del st.session_state.unstable_concept_dict[st.session_state.last_asked_key]
                                  st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                  st.session_state.last_asked_concept = st.session_state.last_asked_key
                                  st.session_state.its_key = True
                                  st.session_state.question_validation = True
                                  print(f"The next key to ask a question about is {st.session_state.last_asked_key} ")
                                  st.session_state.log_buffer.write(
                                      f"The next key to ask a question about is {st.session_state.last_asked_key} \n")
                                  assess_concepts_prompt(concept_name=st.session_state.last_asked_key,
                                                         user_prompt=user_prompt, chat_history=chat_history)
                                  st.session_state.log_buffer.write("\n")
   
   
                              else:  # if there are no other dependencies and keys to ask a question about.
                                  st.session_state.question_validation = False
                                  filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                  sentence = give_advice_sent(filtered_dict)
                                  print(f"The sentence is :  {sentence}")
                                  st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                                  st.session_state.log_buffer.write("\n")
                                  print(st.session_state.users_level)
                                  st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                                  st.session_state.log_buffer.write("\n")
                                  give_advice_users_level(sentence)
                                  st.session_state.unstable_concept_dict = {}
                                  st.session_state.users_level = {}
                                  st.session_state.its_key = True  # not sure if I should do that.
                                  if stop_or_continue(st.session_state.all_concepts) > 3:
                                      st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                      st.session_state.log_buffer.write("\n")
                                      stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
                      else:  # if 'High' in construct name level :
                          st.session_state.users_level[st.session_state.last_asked_key]["dependencies"][
                              st.session_state.last_asked_concept]["status"] = "High"
                          st.session_state.user_high.append(st.session_state.last_asked_concept)
                          st.session_state.unstable_concept_dict[st.session_state.last_asked_key].remove(
                              st.session_state.last_asked_concept)
                          # st.session_state.all_concepts[st.session_state.last_asked_concept]+=1
   
                          if st.session_state.unstable_concept_dict[
                              st.session_state.last_asked_key]:  # if there are other dependencies to ask a q about.
                              random_chosen_concept = random.choice(
                                  st.session_state.unstable_concept_dict[st.session_state.last_asked_key])
                              st.session_state.value = random_chosen_concept
                              st.session_state.last_asked_concept = random_chosen_concept
                              st.session_state.its_key = False
                              st.session_state.question_validation = True
                              st.session_state.log_buffer.write(
                                  f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                              assess_concepts_prompt(concept_name=random_chosen_concept, user_prompt=user_prompt,
                                                     chat_history=chat_history)
                              st.session_state.log_buffer.write("\n")
   
                          else:  # there are no other dependencies to ask a question about
                              del st.session_state.unstable_concept_dict[st.session_state.last_asked_key]
   
                              if st.session_state.unstable_concept_dict:  # if there are other keys to ask a question about.
                                  st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                  st.session_state.last_asked_concept = st.session_state.last_asked_key
                                  st.session_state.its_key = True
                                  st.session_state.question_validation = True
                                  st.session_state.log_buffer.write(
                                      f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                  st.session_state.log_buffer.write(
                                      f"The next key to ask a question about is {st.session_state.last_asked_key}\n")
                                  print(f"The next key to ask a question about is {st.session_state.last_asked_key} ")
                                  assess_concepts_prompt(concept_name=st.session_state.last_asked_key,
                                                         user_prompt=user_prompt,
                                                         chat_history=chat_history)
                                  st.session_state.log_buffer.write("\n")
   
   
                              else:  # if there are no other dependencies to ask a q about.
                                  st.session_state.question_validation = False
                                  st.session_state.its_key = True  # I am not sure if I should have that.
                                  filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                  sentence = give_advice_sent(filtered_dict)
                                  print(f"The sentence is :  {sentence}")
                                  st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                                  print(st.session_state.users_level)
                                  st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                                  give_advice_users_level(sentence)
                                  st.session_state.log_buffer.write("\n")
                                  st.session_state.unstable_concept_dict = {}
                                  st.session_state.users_level = {}
                                  if stop_or_continue(st.session_state.all_concepts) > 3:
                                      st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                      st.session_state.log_buffer.write("\n")
                                      stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
              else:  # if construct_prompt != last_asked_concept :
                  print("Validation in progress...")
                  st.session_state.log_buffer.write("Validation in progress..\n")
                  print(f"{construct_prompt} is not equal to {st.session_state.last_asked_concept}")
                  st.session_state.log_buffer.write(
                      f"{construct_prompt} is not equal to {st.session_state.last_asked_concept}\n")
                  st.session_state.validation_repeat += 1
                  print(f"Validation repeat: {st.session_state.validation_repeat}")
                  st.session_state.log_buffer.write(f"Validation repeat: {st.session_state.validation_repeat}\n")
                  st.session_state.log_buffer.write("\n")
                  if st.session_state.validation_repeat < 3:
                      response = validation_question(last_asked_concept=st.session_state.last_asked_concept,
                                                     chat_history=chat_history)
                      print(response)
                      if "yes" in response:
                          st.session_state.all_concepts[st.session_state.last_asked_concept] += 1
                          print(f"All concepts : {st.session_state.all_concepts}")
                          st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                          st.session_state.question_validation = False
                          st.session_state.validation_repeat = 0
                          print("Question-answer is validated!")
                          st.session_state.log_buffer.write("Question-answer is validated!\n")
                          level = level_check_after_validation(user_answer=chat_history[-1],
                                                               last_asked_concept=st.session_state.last_asked_concept)
                          print(f"After the evaluation the level of the user on the concept is : {level}")
                          st.session_state.log_buffer.write(
                              f"After the evaluation the level of the user on the concept is : {level}\n")
                          st.session_state.log_buffer.write("\n")
                          if "low" in level:
                              if st.session_state.its_key:  # if what we have validated was a key.
                                  st.session_state.users_level[st.session_state.last_asked_concept]["status"] = "Low"
                                  print(f"Low was in the concept, users level are {st.session_state.users_level}")
                                  st.session_state.log_buffer.write(
                                      f"Low was in the concept, users level are {st.session_state.users_level}\n")
                                  st.session_state.log_buffer.write("\n")
                                  if st.session_state.unstable_concept_dict[
                                      st.session_state.last_asked_concept]:  # if the key has dependencies
                                      # ask a question about the dependencies.
                                      st.session_state.its_key = False
                                      st.session_state.question_validation = True
                                      random_chosen_concept = random.choice(
                                          st.session_state.unstable_concept_dict[st.session_state.last_asked_concept])
                                      st.session_state.value = random_chosen_concept
                                      st.session_state.last_asked_concept = random_chosen_concept
                                      st.session_state.log_buffer.write(
                                          f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                      assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                             user_prompt=user_prompt, chat_history=chat_history)
                                      st.session_state.log_buffer.write("\n")
                                  elif not st.session_state.unstable_concept_dict[
                                      st.session_state.last_asked_concept]:  # if there are no dependencies to ask q about
                                      del st.session_state.unstable_concept_dict[st.session_state.last_asked_concept]
                                      # st.session_state.its_key = True #not sure if this should be here
                                      if st.session_state.unstable_concept_dict:  # if there is still something in the unstable concept dict
                                          st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                          st.session_state.last_asked_concept = st.session_state.last_asked_key
                                          st.session_state.its_key = True
                                          st.session_state.question_validation = True
                                          # st.session_state.all_concepts[st.session_state.last_asked_concept] += 1
                                          st.session_state.log_buffer.write(
                                              f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                          assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                                 user_prompt=user_prompt, chat_history=chat_history)
                                          st.session_state.log_buffer.write("\n")
   
                                      else:  # if there are no keys to ask a q about, give advice.
                                          st.session_state.question_validation = False
                                          filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                          sentence = give_advice_sent(filtered_dict)
                                          print(f"The sentence is :  {sentence}")
                                          st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                                          print(st.session_state.users_level)
                                          st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                                          give_advice_users_level(sentence)
                                          st.session_state.log_buffer.write("\n")
                                          st.session_state.unstable_concept_dict = {}
                                          st.session_state.users_level = {}
                                          st.session_state.its_key = True  # not sure if I should do that.
                                          if stop_or_continue(st.session_state.all_concepts) > 3:
                                              st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                              st.session_state.log_buffer.write("\n")
                                              stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
                              else:  # if what we have validated was a value
                                  st.session_state.users_level[st.session_state.last_asked_key]["dependencies"][
                                      st.session_state.last_asked_concept]["status"] = "Low"
                                  st.session_state.unstable_concept_dict[st.session_state.last_asked_key].remove(
                                      st.session_state.last_asked_concept)  # we remove that dependency from the unstable concept dict/
   
                                  if st.session_state.unstable_concept_dict[
                                      st.session_state.last_asked_key]:  # if there are still other dependencies to ask a q about
                                      st.session_state.log_buffer.write(
                                          f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                      random_chosen_concept = random.choice(
                                          st.session_state.unstable_concept_dict[st.session_state.last_asked_key])
                                      st.session_state.its_key = False
                                      st.session_state.last_asked_concept = random_chosen_concept
                                      # st.session_state.all_concepts[random_chosen_concept] += 1
                                      st.session_state.question_validation = True
                                      assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                             user_prompt=user_prompt, chat_history=chat_history)
                                      st.session_state.log_buffer.write("\n")
   
                                  else:  # if there are no other dependencies to ask a q about
                                      del st.session_state.unstable_concept_dict[st.session_state.last_asked_key]
                                      if st.session_state.unstable_concept_dict:  # if there are still other keys to ask a q about
                                          st.session_state.log_buffer.write(
                                              f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                          st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                          st.session_state.last_asked_concept = st.session_state.last_asked_key
                                          st.session_state.its_key = True
                                          st.session_state.question_validation = True
                                          # st.session_state.all_concepts[st.session_state.last_asked_concept] += 1
                                          assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                                 user_prompt=user_prompt, chat_history=chat_history)
                                          st.session_state.log_buffer.write("\n")
                                      else:  # if there are no other keys to ask a q about
                                          st.session_state.question_validation = False
                                          filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                          sentence = give_advice_sent(filtered_dict)
                                          print(f"The sentence is :  {sentence}")
                                          print(st.session_state.users_level)
                                          st.session_state.log_buffer.write(f"The sentence is :  {sentence}")
                                          st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}")
                                          give_advice_users_level(sentence)
                                          st.session_state.log_buffer.write("\n")
                                          st.session_state.unstable_concept_dict = {}
                                          st.session_state.users_level = {}
                                          st.session_state.its_key = True
                                          if stop_or_continue(st.session_state.all_concepts) > 3:
                                              st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                              st.session_state.log_buffer.write("\n")
                                              stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
                          else:  # if "high" in level
                              if st.session_state.its_key:
                                  print(f"We have validated a key and it has a high level! : {level}")
                                  st.session_state.log_buffer.write(
                                      f"We have validated a key and it has a high level! : {level}\n")
                                  st.session_state.log_buffer.write("\n")
                                  st.session_state.user_high.append(st.session_state.last_asked_concept)
                                  st.session_state.users_level[st.session_state.last_asked_concept]["status"] = "High"
                                  del st.session_state.unstable_concept_dict[st.session_state.last_asked_concept]
                                  if st.session_state.unstable_concept_dict:  # if there are other keys to ask a q about.
                                      st.session_state.log_buffer.write(
                                          f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                      st.session_state.its_key = True
                                      st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                      st.session_state.last_asked_concept = st.session_state.last_asked_key
                                      st.session_state.question_validation = True
                                      assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                             user_prompt=user_prompt, chat_history=chat_history)
                                      st.session_state.log_buffer.write("\n")
   
                                  else:  # if there are no other keys to ask a q about
                                      st.session_state.question_validation = False
                                      filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                      sentence = give_advice_sent(filtered_dict)
                                      st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                                      print(f"The sentence is :  {sentence}")
                                      print(st.session_state.users_level)
                                      st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}")
                                      give_advice_users_level(sentence)
                                      st.session_state.log_buffer.write("\n")
                                      st.session_state.unstable_concept_dict = {}
                                      st.session_state.users_level = {}
                                      st.session_state.its_key = True  # not sure about this.
                                      if stop_or_continue(st.session_state.all_concepts) > 3:
                                          st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                          st.session_state.log_buffer.write("\n")
                                          stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
                              else:  # if what we validated with high value was a value and not a key.
                                  print(f"We have validated a value and it has a high level! : {level}")
                                  st.session_state.user_high.append(st.session_state.last_asked_concept)
                                  st.session_state.log_buffer.write(
                                      f"We have validated a value and it has a high level! : {level}\n")
                                  st.session_state.users_level[st.session_state.last_asked_key]["dependencies"][
                                      st.session_state.last_asked_concept]["status"] = "High"
                                  st.session_state.unstable_concept_dict[st.session_state.last_asked_key].remove(
                                      st.session_state.last_asked_concept)
                                  print(f"New unstable concept dict is {st.session_state.unstable_concept_dict}")
                                  print(f"New users level is : {st.session_state.users_level}")
                                  st.session_state.log_buffer.write(
                                      f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                  st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                                  st.session_state.log_buffer.write("\n")
                                  if st.session_state.unstable_concept_dict[
                                      st.session_state.last_asked_key]:  # if there are still other dependencies to ask a q about.
                                      st.session_state.its_key = False
                                      random_chosen_concept = random.choice(
                                          st.session_state.unstable_concept_dict[st.session_state.last_asked_key])
                                      st.session_state.last_asked_concept = random_chosen_concept
                                      st.session_state.question_validation = True
                                      st.session_state.value = st.session_state.last_asked_concept
                                      assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                             user_prompt=user_prompt, chat_history=chat_history)
                                      st.session_state.log_buffer.write("\n")
   
                                  elif not st.session_state.unstable_concept_dict[
                                      st.session_state.last_asked_key]:  # if there are no other dependencies to ask a q about.
                                      st.session_state.its_key = True  # maybe add it to condition below (only if the keys exists)
                                      del st.session_state.unstable_concept_dict[st.session_state.last_asked_key]
                                      if st.session_state.unstable_concept_dict:  # if there are still other keys to ask a q about
                                          st.session_state.log_buffer.write(
                                              f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                          st.session_state.its_key = True
                                          st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                          st.session_state.last_asked_concept = st.session_state.last_asked_key
                                          st.session_state.question_validation = True
                                          assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                                 user_prompt=user_prompt, chat_history=chat_history)
                                          st.session_state.log_buffer.write("\n")
   
                                      else:  # if there is nothing to ask a q about.
                                          st.session_state.question_validation = False
                                          filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                          sentence = give_advice_sent(filtered_dict)
                                          print(f"The sentence is :  {sentence}")
                                          print(st.session_state.users_level)
                                          st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                                          st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                                          give_advice_users_level(sentence)
                                          st.session_state.log_buffer.write("\n")
                                          st.session_state.unstable_concept_dict = {}
                                          st.session_state.users_level = {}
                                          st.session_state.its_key = True  # not sure about this.
                                          if stop_or_continue(st.session_state.all_concepts) > 3:
                                              st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                              st.session_state.log_buffer.write("\n")
                                              stop_button = st.button("Stop the conversation.", on_click=stop_button)
                      else:  # if 'no' in response
                          clarification_question(last_asked_concept=st.session_state.last_asked_concept,
                                                 chat_history=chat_history)
   
                  else:  # if validation repeat >3
                      st.session_state.validation_repeat = 0
   
                      if st.session_state.its_key:
                          print("Tried to validate the key but it has been asked more than twice.")
                          st.session_state.log_buffer.write(
                              "Tried to validate the key but it has been asked more than twice.\n")
                          if st.session_state.unstable_concept_dict[
                              st.session_state.last_asked_key]:  # if there are dependencies in the key to ask a q about.
                              st.session_state.its_key = False
                              st.session_state.log_buffer.write(
                                  f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                              random_chosen_concept = random.choice(
                                  st.session_state.unstable_concept_dict[st.session_state.last_asked_concept])
                              print(
                                  f"Could not validate {st.session_state.last_asked_key} hence validating its dependency : {random_chosen_concept}")
                              st.session_state.log_buffer.write(
                                  f"Could not validate {st.session_state.last_asked_key} hence validating its dependency : {random_chosen_concept}\n")
                              st.session_state.last_asked_concept = random_chosen_concept
                              st.session_state.value = st.session_state.last_asked_concept
                              st.session_state.question_validation = True
                              assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                     user_prompt=user_prompt, chat_history=chat_history)
                              st.session_state.log_buffer.write("\n")
   
                          else:  # if there are no dependencies.
                              print("There were no dependencies of the key hence the key is being removed.")
                              st.session_state.log_buffer.write(
                                  "There were no dependencies of the key hence the key is being removed.\n")
                              del st.session_state.unstable_concept_dict[st.session_state.last_asked_key]
                              if st.session_state.unstable_concept_dict:  # if there are still other keys to ask a q about.
                                  st.session_state.log_buffer.write(
                                      f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                  st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                  st.session_state.last_asked_concept = st.session_state.last_asked_key
                                  st.session_state.its_key = True
                                  st.session_state.question_validation = True
                                  print(f"The new key to ask a question about is {st.session_state.last_asked_key} ")
                                  st.session_state.log_buffer.write(
                                      f"The new key to ask a question about is {st.session_state.last_asked_key}\n")
                                  assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                         user_prompt=user_prompt, chat_history=chat_history)
                                  st.session_state.log_buffer.write("n")
   
                              else:  # There are no other keys to ask a q about so give advice
                                  st.session_state.question_validation = False
                                  filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                  sentence = give_advice_sent(filtered_dict)
                                  print(f"The sentence is :  {sentence}")
                                  st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                                  print(st.session_state.users_level)
                                  st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
   
                                  give_advice_users_level(sentence)
                                  st.session_state.log_buffer.write("\n")
                                  st.session_state.unstable_concept_dict = {}
                                  st.session_state.users_level = {}
                                  st.session_state.its_key = True  # not sure about this.
                                  if stop_or_continue(st.session_state.all_concepts) > 3:
                                      st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                      st.session_state.log_buffer.write("\n")
                                      stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
                      else:  # if not its.key so its a value
                          st.session_state.unstable_concept_dict[st.session_state.last_asked_key].remove(
                              st.session_state.last_asked_concept)
                          # delete that value from the unstable concept dict
                          if st.session_state.unstable_concept_dict[
                              st.session_state.last_asked_key]:  # if there are other dependencies to ask a q about.
                              st.session_state.log_buffer.write(
                                  f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                              st.session_state.its_key = False
                              random_chosen_concept = random.choice(
                                  st.session_state.unstable_concept_dict[st.session_state.last_asked_key])
                              st.session_state.last_asked_concept = random_chosen_concept
                              st.session_state.value = st.session_state.last_asked_concept
                              st.session_state.question_validation = True
                              # st.session_state.all_concepts[st.session_state.last_asked_concept] += 1
                              print(f"Previous value is removed new chosen value is : {st.session_state.last_asked_concept}")
                              st.session_state.log_buffer.write(
                                  f"Previous value is removed new chosen value is : {st.session_state.last_asked_concept}")
                              assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                     user_prompt=user_prompt, chat_history=chat_history)
                              st.session_state.log_buffer.write("\n")
   
                          else:  # if there are no other dependencies to ask a q about.
                              del st.session_state.unstable_concept_dict[st.session_state.last_asked_key]
                              if st.session_state.unstable_concept_dict:  # if there are other keys to ask a q about.
                                  st.session_state.log_buffer.write(
                                      f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                                  st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                                  st.session_state.last_asked_concept = st.session_state.last_asked_key
                                  # st.session_state.all_concepts[st.session_state.last_asked_concept] += 1
                                  st.session_state.its_key = True
                                  st.session_state.question_validation = True
                                  assess_concepts_prompt(concept_name=st.session_state.last_asked_concept,
                                                         user_prompt=user_prompt, chat_history=chat_history)
                                  st.session_state.log_buffer.write("\n")
   
                              else:  # if there are no other keys to ask a q about.
                                  st.session_state.question_validation = False
                                  filtered_dict = filter_low_dependencies_as_list(st.session_state.users_level)
                                  sentence = give_advice_sent(filtered_dict)
                                  print(f"The sentence is :  {sentence}")
                                  st.session_state.log_buffer.write(f"The sentence is :  {sentence}\n")
                                  st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                                  print(st.session_state.users_level)
                                  give_advice_users_level(sentence)
                                  st.session_state.log_buffer.write("\n")
                                  st.session_state.unstable_concept_dict = {}
                                  st.session_state.users_level = {}
                                  st.session_state.its_key = True  # not sure about this.
                                  if stop_or_continue(st.session_state.all_concepts) > 3:
                                      st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                      st.session_state.log_buffer.write("\n")
                                      stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
          else:  # if question validation is not necessary
              if concept_level == "No concept":
                  lowest_concept = choose_lowest_concept(st.session_state.all_concepts)
                  st.session_state.log_buffer.write(f"Chosen lowest concept : {lowest_concept}\n")
                  if st.session_state.asked_concepts.count(lowest_concept) > 2:
                      st.session_state.all_concepts[lowest_concept] += 1
                      st.session_state.log_buffer.write(f"Previous asked concepts : {st.session_state.asked_concepts}\n")
                      st.session_state.asked_concepts = [i for i in st.session_state.asked_concepts if i != lowest_concept]
                      st.session_state.log_buffer.write(f"New asked concepts : {st.session_state.asked_concepts}\n")
                      lowest_concept = choose_lowest_concept(st.session_state.all_concepts)
                      st.session_state.log_buffer.write(f"New chosen lowest concept : {lowest_concept}\n")
                      st.session_state.asked_concepts.append(lowest_concept)
                      if st.session_state.advice_given:
                         assess_concepts_prompt(lowest_concept, user_prompt=user_prompt, chat_history=chat_history)
                         # st.session_state.all_concepts[lowest_concept]+=1
                         print(f"Concept level : {st.session_state.all_concepts}")
                         st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                         st.session_state.log_buffer.write("\n")
                         if stop_or_continue(st.session_state.all_concepts) > 3:
                             st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                             st.session_state.log_buffer.write("\n")
                             stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
                      else:
                         if stop_or_continue(st.session_state.all_concepts) > 3:
                            sentence = high_level_praise_advice(st.session_state.user_high,concept_definitions_dict)
                            generic_advice_assess_concepts(sentence,concept_name=lowest_concept,user_prompt=user_prompt,chat_history=chat_history)
                            st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                            st.session_state.log_buffer.write("\n")
                            stop_button = st.button("Stop the conversation.", on_click=stop_button)
                             
                         else:
                            assess_concepts_prompt(lowest_concept, user_prompt=user_prompt, chat_history=chat_history)
                            st.session_state.asked_concepts.append(lowest_concept)
                            st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                            st.session_state.log_buffer.write("\n")
                           
                        
                        
                  else:
                      assess_concepts_prompt(lowest_concept, user_prompt=user_prompt, chat_history=chat_history)
                      st.session_state.asked_concepts.append(lowest_concept)
                      st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                      st.session_state.log_buffer.write("\n")
                      if stop_or_continue(st.session_state.all_concepts) > 3:
                         st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                         st.session_state.log_buffer.write("\n")
                         stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
   
              else:  # If concept level equals to one of the concepts.
                  print(f"The identified concept is {concept_level}")
                  if "Low" in construct_name_level:
                      st.session_state.main_problem_concept = concept_level
                      st.session_state.all_concepts[st.session_state.main_problem_concept] += 1
                      st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                      print(f"All concepts: {st.session_state.all_concepts}")
                      st.session_state.unstable_concept_dict = retrieve_concepts(st.session_state.main_problem_concept)
                      print(f"The unstable concept dictionary is {st.session_state.unstable_concept_dict}")
                      st.session_state.log_buffer.write(
                          f"UNSTABLE CONCEPT DICTIONARY : {st.session_state.unstable_concept_dict}\n")
                      st.session_state.log_buffer.write("\n")
                      if not st.session_state.unstable_concept_dict:  # if the dictionary is empty, do not even create users_level and just give advice then empty the unstable concept dict
                          print(
                              "The dictionary was empty. Hence we don't need to create a users level dict. So we will give advice!")
                          st.session_state.log_buffer.write(
                              "The dictionary was empty. Hence we don't need to create a users level dict. So we will give advice!\n")
                          st.session_state.log_buffer.write("\n")
                          give_advice_prompt(main_problem=st.session_state.main_problem_concept, user_prompt=user_prompt,
                                             chat_history=chat_history)
                          st.session_state.unstable_concept_dict = {}
                          if stop_or_continue(st.session_state.all_concepts) > 3:
                              st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                              st.session_state.log_buffer.write("\n")
                              stop_button = st.button("Stop the conversation.", on_click=stop_button)
                      else:  # If the dictionary is not empty and hence there are has_effect_relations to further ask a question about.
                          st.session_state.users_level = initialize_users_level_dict(st.session_state.unstable_concept_dict)
                          print(f"The users level dictionary : {st.session_state.users_level}")
                          st.session_state.log_buffer.write(f"USERS LEVEL : {st.session_state.users_level}\n")
                          # Now ask a question about the first key in that dictionary.
                          st.session_state.last_asked_key = next(iter(st.session_state.unstable_concept_dict))
                          st.session_state.its_key = True
                          print(f"First key/concept to ask a question about is {st.session_state.last_asked_key}")
                          st.session_state.log_buffer.write(
                              f"First key/concept to ask a question about is {st.session_state.last_asked_key}\n")
                          # st.session_state.all_concepts[st.session_state.last_asked_key] += 1
                          st.session_state.last_asked_concept = st.session_state.last_asked_key
                          st.session_state.question_validation = True
                          assess_concepts_prompt(concept_name=st.session_state.last_asked_key, user_prompt=user_prompt,
                                                 chat_history=chat_history)
   
                  else:  # if "High" in construct name level
                      st.session_state.main_problem_concept = concept_level
                      st.session_state.all_concepts[st.session_state.main_problem_concept] += 1
                      st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                      print(f"ALL CONCEPTS : {st.session_state.all_concepts}")
                      print(f"The user had a high value for the concept {st.session_state.main_problem_concept}")
                      st.session_state.user_high.append(st.session_state.main_problem_concept)
                      st.session_state.log_buffer.write(
                          f"The user had a high value for the concept {st.session_state.main_problem_concept}\n")
                      lowest_concept = choose_lowest_concept(st.session_state.all_concepts)
                      if st.session_state.asked_concepts.count(lowest_concept) > 2:
                          st.session_state.all_concepts[lowest_concept] += 1
                          st.session_state.log_buffer.write(f"Previous asked concepts : {st.session_state.asked_concepts}\n")
                          st.session_state.asked_concepts = [i for i in st.session_state.asked_concepts if
                                                             i != lowest_concept]
                          st.session_state.log_buffer.write(f"New asked concepts : {st.session_state.asked_concepts}\n")
                          lowest_concept = choose_lowest_concept(st.session_state.all_concepts)
                          st.session_state.asked_concepts.append(lowest_concept)
                          st.session_state.log_buffer.write(f"New chosen lowest concept : {lowest_concept}\n")
                          if st.session_state.advice_given:
                             assess_concepts_prompt(lowest_concept, user_prompt=user_prompt, chat_history=chat_history)
                             st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                             st.session_state.log_buffer.write("\n")
                             if stop_or_continue(st.session_state.all_concepts) > 3:
                                 st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                 st.session_state.log_buffer.write("\n")
                                 stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
                          else:
                             if stop_or_continue(st.session_state.all_concepts) > 3:
                                sentence = high_level_praise_advice(st.session_state.user_high,concept_definitions_dict)
                                generic_advice_assess_concepts(sentence,concept_name=lowest_concept,user_prompt=user_prompt,chat_history=chat_history)
                                st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                                st.session_state.log_buffer.write("\n")
                                stop_button = st.button("Stop the conversation.", on_click=stop_button)
                                
                             else:
                                assess_concepts_prompt(lowest_concept, user_prompt=user_prompt, chat_history=chat_history)
                                st.session_state.asked_concepts.append(lowest_concept)
                                st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                                st.session_state.log_buffer.write("\n")
   
                          
                      else:
                         if st.session_state.advice_given:
                            assess_concepts_prompt(lowest_concept, user_prompt=user_prompt, chat_history=chat_history)
                            st.session_state.asked_concepts.append(lowest_concept)
                            st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                            st.session_state.log_buffer.write("\n")
                            if stop_or_continue(st.session_state.all_concepts) > 3:
                               st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                               st.session_state.log_buffer.write("\n")
                               stop_button = st.button("Stop the conversation.", on_click=stop_button)
   
                         else:
                            if stop_or_continue(st.session_state.all_concepts) > 3:
                               sentence = high_level_praise_advice(st.session_state.user_high,concept_definitions_dict)
                               generic_advice_assess_concepts(sentence,concept_name=lowest_concept,user_prompt=user_prompt,chat_history=chat_history)
                               st.session_state.log_buffer.write("Stop the conversation button presented.\n")
                               st.session_state.log_buffer.write("\n")
                               stop_button = st.button("Stop the conversation.", on_click=stop_button)
                               
                            else:
                               assess_concepts_prompt(lowest_concept, user_prompt=user_prompt, chat_history=chat_history)
                               st.session_state.asked_concepts.append(lowest_concept)
                               st.session_state.log_buffer.write(f"ALL CONCEPTS : {st.session_state.all_concepts}\n")
                               st.session_state.log_buffer.write("\n")

elif st.session_state.experiment_condition == 2:

   st.write("CONTROL CONDITION")

   if "user_id" not in st.session_state:
      st.session_state.user_id =("CTRL_" + (UUIDShortener.encode(str(uuid.uuid4()))))


   if st.session_state.start_experiment == "experiment":


      for message in st.session_state["messages"]:
             with st.chat_message(message["role"]):
                 st.markdown(message["content"])
      
      
      if user_prompt := st.chat_input("Want to share some thoughts?"):
         st.session_state.messages.append({"role": "user", "content": user_prompt})
         st.session_state.log_buffer.write(f"USER SAID : {user_prompt}\n")
         st.session_state.log_buffer.write("\n")
      
         with st.chat_message("user"):
            st.markdown(user_prompt)

      chat_history = [(message["role"], message["content"]) for message in st.session_state.messages]

      if user_prompt:
         completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
               {"role":"assistant",
                "content":f"Your goal is to help people become more physically active. Therefore, taking the user input: {user_prompt} and the chat history : {chat_history} into account, answer the user and, if appropriate, provide advice. Aim to steer the conversation naturally toward physical activity."
                  
               }
            ]
         )
   
         answer = completion.choices[0].message.content
   
   
         with st.chat_message("assistant"):
            response = st.write_stream(generate_response(answer))
            st.session_state.messages.append({"role":"assistant","content":response})


         if len(chat_history) > 1:
            st.session_state.log_buffer.write("Stop the conversation button presented.\n")
            st.session_state.log_buffer.write("\n")
            stop_button = st.button("Stop the conversation.", on_click=stop_button)
            
            
                              


if st.session_state.start_experiment == "post-survey":
   st.write("To finalize your participation, please complete the short survey below. Providing honest responses is essential to ensure your contributions are accurately reflected in the study. The scale ranges from strongly disagree to strongly agree, including options disagree, somewhat disagree, neutral, somewhat agree, agree, and strongly agree.\n")
   question1 = survey.select_slider("Q1 : The chatbot personalized the conversation based on my personal information.", options=["Strongly Disagree","Disagree","Somewhat Disagree","Neutral","Somewhat Agree","Agree","Strongly Agree"],id="Q1")
   question2 = survey.select_slider("Q2 : I am satisfied with the advice given to me by this chatbot.",options=["Strongly Disagree","Disagree","Somewhat Disagree","Neutral","Somewhat Agree","Agree","Strongly Agree"],id="Q2")
   question3 = survey.select_slider("Q3 : The chatbot used my thoughts and habits regarding physical activity to provide personalized advice.",options=["Strongly Disagree","Disagree","Somewhat Disagree","Neutral","Somewhat Agree","Agree","Strongly Agree"],id="Q3")
   question4 = survey.select_slider("Q4 : I would recommend this chatbot to a friend who wants help with improving their physical activity.",options =["Strongly Disagree","Disagree","Somewhat Disagree","Neutral","Somewhat Agree","Agree","Strongly Agree"],id = "Q4")
   submit_button = st.button("Submit",on_click=post_survey_submit)


if st.session_state.start_experiment == "stop-experiment":
   st.empty()
   st.markdown(" ## **Thank you for participating!**")




if not st.session_state.uploaded_to_bucket:
   if st.session_state.save_conversation:
      upload_to_bucket("phy_assistant_bucket", st.session_state.log_buffer.getvalue(), user_id=st.session_state.user_id)













