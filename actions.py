from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from db import insert_data
import sqlite3
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
import torch
import re
import numpy as np
from typing import Optional
import requests
from bs4 import BeautifulSoup

class ActionProcessStudentPreferences(Action):
    def name(self):
        return "action_process_student_preferences"

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        self.known_domains = {
            "Web Development": "Web development involves creating and maintaining websites. It includes aspects like web design, web publishing, web programming, and database management.",
            "Machine Learning": "Machine learning is a type of artificial intelligence (AI) that allows software applications to become more accurate in predicting outcomes without being explicitly programmed to do so.",
            "Cloud Computing": "Cloud computing is the delivery of different services through the Internet, including data storage, servers, databases, networking, and software.",
            "Software Development": "Software development is the process of conceiving, specifying, designing, programming, documenting, testing, and bug fixing involved in creating and maintaining applications, frameworks, or other software components.",
            "Quantum Computing": "Quantum computing uses the principles of quantum mechanics to process information. It represents and manipulates information using quantum bits, or qubits, which can be in multiple states simultaneously, offering vast computational power.",
            "Deep Learning": "Deep learning is a subset of machine learning in artificial intelligence that has networks capable of learning unsupervised from data that is unstructured or unlabeled. It involves neural networks with three or more layers.",
            "Data Science": "Data science combines multiple fields, including statistics, scientific methods, and data analysis, to extract value from data. It encompasses preparing data for analysis, computing various statistics, and analyzing the data to make predictions."
        }
        self.domain_embeddings = self.generate_domain_embeddings()

    def generate_domain_embeddings(self):
        domain_embeddings = {}
        for domain, description in self.known_domains.items():
            inputs = self.tokenizer(description, return_tensors="pt", truncation=True, max_length=512, padding=True)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(1).detach().numpy()
            domain_embeddings[domain] = embeddings[0]
        return domain_embeddings

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict):
        fav_module = tracker.get_slot('fav_module_slot')
        least_fav_module = tracker.get_slot('least_fav_module_slot')
        interested_domain = tracker.get_slot('interested_domain_slot')
        not_interested_domain = tracker.get_slot('not_interested_domain_slot')

        finalized_domain = self.process_student_inputs(
            fav_module, least_fav_module, interested_domain, not_interested_domain)

        slots = [SlotSet("preferred_domain", finalized_domain)]

        dispatcher.utter_message(text=f"Based on your preferences, I recommend focusing on the {finalized_domain} domain.")

        return slots

    def embed_input(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(1).detach().numpy()
        return embeddings[0]

    def find_closest_domain(self, preference_embedding):
        closest_domain = None
        highest_similarity = float('-inf')
        for domain, embedding in self.domain_embeddings.items():
            similarity = 1 - cosine(preference_embedding, embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_domain = domain
        return closest_domain

    def process_student_inputs(self, favorite_module, least_favorite_module, interested_domain, not_interested_domain):
        preference_description = " ".join(filter(None, [favorite_module, least_favorite_module, interested_domain, not_interested_domain]))
        if preference_description:
            preference_embedding = self.embed_input(preference_description)
            finalized_domain = self.find_closest_domain(preference_embedding)
        else:
            finalized_domain = "General Domain"
        return finalized_domain

class SemanticAnalysisActionintdom(Action):
    def name(self):
        return "action_semantic_analysis_not_favourite_module"

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def run(self, dispatcher, tracker, domain):
        last_message = tracker.latest_message.get('text')

        entity_value = self.extract_entities(last_message)

        slots = [SlotSet("least_fav_module_slot", entity_value)]
        dispatcher.utter_message(text="Which domain you are intrested")

        return slots

    def extract_entities(self, text):

        patterns = [
             r"not interested in \[?([^]]+)\]?",
                r"i don't like \[?([^]]+)\]?",
                r"my least favourite is \[?([^]]+)\]?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

class SemanticAnalysisActionintdom(Action):
    def name(self):
        return "action_semantic_analysis_favourite_module"

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def run(self, dispatcher, tracker, domain):
        last_message = tracker.latest_message.get('text')


        entity_value = self.extract_entities(last_message)

        slots = [SlotSet("fav_module_slot", entity_value)]
        dispatcher.utter_message(text="Which module you are least interested")

        return slots

    def extract_entities(self, text):

        patterns = [
            r"my favourite module is \[?([^]]+)\]?",
            r"i love \[?([^]]+)\]?",
            r"i like \[?([^]]+)\]?",
            r"favourite is \[?([^]]+)\]?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

class SemanticAnalysisActionintdom(Action):
    def name(self):
        return "action_semantic_analysis_interested_domain"

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def run(self, dispatcher, tracker, domain):
        last_message = tracker.latest_message.get('text')

        entity_value = self.extract_entities(last_message)

        slots = [SlotSet("interested_domain_slot", entity_value)]
        dispatcher.utter_message(text="Which domain you are least interested")

        return slots

    def extract_entities(self, text):

        patterns = [
            r"interested in \[?([^]]+)\]?",
            r"i want to explore \[?([^]]+)\]?",
            r"very much interested in \[?([^]]+)\]?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

class ActionSemanticAnalysisKnownDomain(Action):
    def name(self):
        return "action_extract_known_domain"

    def run(self, dispatcher, tracker, domain):
        last_message = tracker.latest_message.get('text')

        known_domain_entity = self.extract_known_domain_entity(last_message)

        if known_domain_entity:
            slots = [SlotSet("preferred_domain", known_domain_entity)]
            dispatcher.utter_message(text=f"Would you like to know about Supervisors in {known_domain_entity}?")
        else:
            dispatcher.utter_message(text="Could you specify which domain you are interested in?")
            slots = []

        return slots

    def extract_known_domain_entity(self, text):
        pattern = r"I want to do my dissertation on ([^\,\.]+?)(?: but|,|\.|$)|my topic ([^\,\.]+?)(?: but|,|\.|$)|lectures available in ([^\,\.]+?)(?: but|,|\.|$)"

        matches = re.findall(pattern, text, re.IGNORECASE)
        matches = [match for sublist in matches for match in sublist if match]
        if matches:
            return matches[0]

        return None


class SemanticAnalysisAction(Action):
    def name(self):
        return "action_semantic_analysis_not_interested_domain"

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def run(self, dispatcher, tracker, domain):
        last_message = tracker.latest_message.get('text')

        entity_value = self.extract_entities(last_message)

        slots = [SlotSet("not_interested_domain_slot", entity_value)]
        insert_data(tracker.get_slot('cn'), tracker.get_slot('dn'), tracker.get_slot('en'), tracker.get_slot('fn'),
                    tracker.get_slot('fav_module_slot'), tracker.get_slot('least_fav_module_slot'), tracker.get_slot('interested_domain_slot'), tracker.get_slot('not_interested_domain_slot'))


        return slots

    def extract_entities(self, text):

        patterns = [
            r"not interested in \[?([^]]+)\]?",
            r"([^]]+) does not interest me",
            r"not interested is \[?([^]]+)\]?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None


class ActionHelloWorld(Action):

     def name(self) -> Text:
         return "action_student_modules"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
         a= next(tracker.get_latest_entity_values('c'),None)
         b = next(tracker.get_latest_entity_values('d'), None)
         c = next(tracker.get_latest_entity_values('e'), None)
         d = next(tracker.get_latest_entity_values('f'), None)
         l=[]
         if a:
             l.append(SlotSet('cn',a))
         if b:
             l.append(SlotSet('dn',b))
         if c:
             l.append(SlotSet('en',c))
         if d:
             l.append(SlotSet('fn',d))


         dispatcher.utter_message(text="From the above modules What's your favourite module")


         return l



class Actionout(Action):

    def name(self) -> Text:
        return "action_student_interested_domain"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()
        cursor.execute('''SELECT lecture_1, lecture_2, lecture_3 FROM lecture_test''')
        lecture_details = cursor.fetchone()
        f = tracker.get_slot('in')
        if lecture_details:
            lecture_1, lecture_2, lecture_3 = lecture_details
            dispatcher.utter_message(
                text=f"For a project that intersects {f}, you have some excellent Lectures at our university. \nLecture details: \nLecture 1: {lecture_1}\nLecture 2: {lecture_2}\nLecture 3: {lecture_3}\n")
        else:
            dispatcher.utter_message(text="No lecture details found in the database.")

        conn.close()

        return []

class Actionout(Action):

    def name(self) -> Text:
        return "action_lecture_research_paper"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()
        cursor.execute('''SELECT lec_1_res, lec_2_res, lec_3_res FROM lecture_research_test''')
        lecture_details = cursor.fetchone()

        if lecture_details:
            lecture_1, lecture_2, lecture_3 = lecture_details
            dispatcher.utter_message(
                text=f"Here you go! \nLecture Research details: \n{lecture_1}\n{lecture_2}\n{lecture_3}\n")
        else:
            dispatcher.utter_message(text="No lecture research details found in the database.")

        conn.close()

        return []

class Actionknowd(Action):

    def name(self) -> Text:
        return "action_known_domain"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('domain_name'), None)
        l=[]
        if e:
            l.append(SlotSet('known_domain_slot','e'))
        dispatcher.utter_message(text="Okay These are the lectures")
        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()
        cursor.execute('''SELECT lecture_1, lecture_2, lecture_3 FROM lecture_test''')
        lecture_details = cursor.fetchone()
        f = tracker.get_slot('known_domain_slot')
        if lecture_details:
            lecture_1, lecture_2, lecture_3 = lecture_details
            dispatcher.utter_message(
                text=f"For a project that intersects {f}, you have some excellent options at our university. \nLecture details: \nLecture 1: {lecture_1}\nLecture 2: {lecture_2}\nLecture 3: {lecture_3}\n")
        else:
            dispatcher.utter_message(text="No lecture details found in the database.")

        conn.close()

        return l

class ActionFav(Action):

    def name(self) -> Text:
        return "action_fav"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('module_name'), None)
        l=[]
        if e:
            l.append(SlotSet('fav_module_slot','e'))
        dispatcher.utter_message(text="Which module you are least intrested")

        return l
class ActionLeastFav(Action):

    def name(self) -> Text:
        return "action_least_fav"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('module_name'), None)
        l=[]
        if e:
            l.append(SlotSet('least_fav_module_slot','e'))
        dispatcher.utter_message(text="Which domain you are intrested")

        return l
class Actionintdom(Action):

    def name(self) -> Text:
        return "action_int_domain"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('domain_name'), None)
        l=[]
        if e:
            l.append(SlotSet('interested_domain_slot','e'))

        dispatcher.utter_message(text="Which domain you are least intrested")

        return l
class Actionnotintdom(Action):

    def name(self) -> Text:
        return "action_not_int_domain"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('domain_name'), None)
        l=[]
        if e:
            l.append(SlotSet('not_interested_domain_slot','e'))
        insert_data(tracker.get_slot('cn'), tracker.get_slot('dn'), tracker.get_slot('en'), tracker.get_slot('fn'), tracker.get_slot('fav_module_slot'), tracker.get_slot('least_fav_module_slot'), tracker.get_slot('interested_domain_slot'), tracker.get_slot('not_interested_domain_slot'))

        return l


class ActionFetchAndDisplayLectures(Action):
    def name(self):
        return "action_fetch_and_display_lectures"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        domain_name = tracker.get_slot('preferred_domain')
        print(f"Retrieved preferred_domain: {domain_name}")
        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()
        query = "SELECT name FROM lecture WHERE lower(domain) = lower(?)"
        cursor.execute(query, (domain_name,))
        lectures = cursor.fetchall()
        conn.close()
        if lectures:
            lecture_names = [name[0] for name in lectures]
            lecture_details = "\n".join(lecture_names)
            dispatcher.utter_message(text=f"Supervisors in the domain of {domain_name}:\n{lecture_details}")
        else:
            dispatcher.utter_message(text=f"No Supervisors found in the domain of {domain_name}.")

        return []


class ActionFetchAndDisplayLecturesResearch(Action):
    def name(self):
        return "action_fetch_and_display_lectures_research"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        domain_name = tracker.get_slot('preferred_domain')

        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()

        query = "SELECT research FROM lecture WHERE lower(domain) = lower(?)"

        cursor.execute(query, (domain_name,))
        lectures = cursor.fetchall()
        conn.close()
        if lectures:
            lecture_names = [research[0] for research in lectures]  # Extracts the first element from each tuple
            lecture_details = "\n".join(lecture_names)  # Joins the lecture names into a single string
            dispatcher.utter_message(text=f"Research Work of Supervisors in the domain of {domain_name}:\n{lecture_details}")
        else:
            dispatcher.utter_message(text=f"No Supervisors Research Work found in the domain of {domain_name}.")

        return []

        return []


class ActionScrapeYanchao(Action):

    def name(self) -> Text:
        return "action_scrape_yanchao"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        url = "https://www.napier.ac.uk/people/yanchao-yu"
        try:
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            content = soup.find("div", id="tab-1").find("p").text
            dispatcher.utter_message(text=content)
        except Exception as e:
            dispatcher.utter_message(text="Could not retrieve information about Yanchao.")

        return []

class ActionScrapePersonBio(Action):

    def name(self) -> Text:
        return "action_scrape_person_bio"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        person_name_entity = next(tracker.get_latest_entity_values("person"), None)
        if person_name_entity:
            search_name = re.match(r"(\w+)", person_name_entity).group(0).lower()

            valid_names = {
                "yanchao": "yanchao-yu",
                "gordon": "gordon-russell",
                "craig": "craig-thomson",
                "amjad": "amjad-ullah",
                "berk": "berk-canberk",
                "amir": "amir-hussain",
            }

            url_name = valid_names.get(search_name)

            if url_name:
                url = f"https://www.napier.ac.uk/people/{url_name}"

                try:
                    page = requests.get(url)
                    soup = BeautifulSoup(page.content, "html.parser")
                    content = soup.find("div", id="tab-1").find("p").text
                    dispatcher.utter_message(text=content)
                except Exception as e:
                    dispatcher.utter_message(text=f"Could not retrieve information for {person_name_entity}.")
            else:
                dispatcher.utter_message(text=f"Sorry, I don't have information on {person_name_entity}.")
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find the name you're asking about.")

        return []

class ActionTestingnames(Action):

    def name(self) -> Text:
        return "action_testing_names"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('domain_name'), None)
        l=[]
        a= tracker.get_slot('fav_module_slot')
        b= tracker.get_slot('least_fav_module_slot')
        c= tracker.get_slot('interested_domain_slot')
        d= tracker.get_slot('not_interested_domain_slot')
        dispatcher.utter_message(text=f"Your input domains, {a}, {b}, {c},{d} .")

        return l
