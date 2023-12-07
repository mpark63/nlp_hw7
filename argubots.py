"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `eval.py`.
We've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo

# Use the same logger as agents.py, since argubots are agents;
# we split this file  
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files


###########################################
# Define your own additional argubots here!
###########################################


class AkikiAgent(KialoAgent):
    """ AkikiAgent subclasses the KialoAgent class. It responds with a relevant claim from
    a Kialo database. No LLM is used. It's better at answering short queries than Akiko."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:
        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            """
            * include earlier dialogue turns in the BM25 query only if the BM25 similarity is too low without them
            * weight more recent turns more heavily in the BM25 query (how can you arrange that?)
            * treat the human's earlier turns differently from Akiki's own previous turns
            """
            # previous_user_turn = d[-1]['content']  # previous turn from user
            # previous_bot_turn = d[-2]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            previous_user_turns = [d[i]['content']*(i+1) for i in range(len(d)) if d[i]['speaker'] != self.name]
            previous_user_turns = " ".join(previous_user_turns)

            previous_bot_turns = [d[i]['content']*(i+1) for i in range(len(d)) if d[i]['speaker'] == self.name]
            previous_bot_turns = " ".join(previous_bot_turns)

            neighbors = self.kialo.closest_claims(previous_user_turns, n=3, kind='has_cons', threshold=8)
            if len(neighbors) == 0 and len(d) > 1: 
                neighbors = self.kialo.closest_claims(previous_bot_turns, n=3, kind='has_pros', threshold=8)
            if len(neighbors) == 0: 
                # no threshold 
                neighbors = self.kialo.closest_claims(previous_user_turns, n=3, kind='has_cons')
            
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor] or self.kialo.pros[neighbor])
        
        return claim
    
akiki = AkikiAgent("Akiki", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files
