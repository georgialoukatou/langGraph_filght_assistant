import getpass
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import openai
import uuid
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from agent_tools import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket, lookup_policy
from helper_functions import update_dates
import os
import sqlite3
import pandas as pd
import requests
from typing import Annotated
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode
from datetime import date, datetime
from typing import Optional

import pytz
from langchain_core.runnables import RunnableConfig
import os

import sqlite3

import pandas as pd
import requests

import shutil
import sqlite3
import pandas as pd
import requests
from typing import Annotated
import re

import numpy as np
import openai
from langchain_core.tools import tool

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
load_dotenv()

import streamlit as st
import uuid
#from assistant import part_1_graph  # Import the graph from your assistant code
from langchain_core.messages import ToolMessage

st.set_page_config(page_title="Swiss Airlines Customer Assistant", layout="wide")
st.title("✈️ Swiss Airlines Customer Assistant")




def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            print("Response: " + str(message))
            msg_repr = message.pretty_repr(html=False)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
            msg_content = getattr(message, "content", msg_repr)
            return msg_content



class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
# The backup lets us restart for each tutorial section
backup_file = "travel2.backup.sqlite"
overwrite = False
if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(local_file, backup_file)

db = update_dates(local_file)


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

from datetime import date, datetime

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


part_1_tools = [
    lookup_policy,
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,

]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

#from IPython.display import Image, display

#try:
#    display(Image(part_1_graph.get_graph(xray=True).draw_mermaid_png()))
#except Exception:
#    # This requires some extra dependencies and is optional
#    pass

# Let's create an example conversation a user might have with the assistant
tutorial_questions = ['Hello, can I get an invoice for my flight?', 'What is the policy if I want to get a refund?', 'I changed my mind, when exactly is my flight? Is there a way to change it for the next one?']


# Update with the backup file so we can restart from the original place in each section
#db = update_dates(db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

_printed = set()

#for question in tutorial_questions:
#    events = part_1_graph.stream(
#        {"messages": ("user", question)}, config, stream_mode="values"
#    )
#    for event in events:
#        _print_event(event, _printed)


# Streamlit app configuration
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []  # List to store the chat history

# Chat input
user_input = st.chat_input("How can I assist you today?")

if user_input:

    # Append the user's message to the conversation history
    st.session_state["conversation_history"].append({"role": "user", "content": user_input})

    # Stream events and collect responses
    events = part_1_graph.stream(
        {"messages": ("user", user_input)}, config, stream_mode="values"
    )


    # Placeholder for the final assistant response
    assistant_response = ""

    # Process events and extract response
    for event in events:
        output = _print_event(event, _printed)
        if output:  # Check if there is valid content
            assistant_response = output  # Capture the last response

    # Display the user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display the assistant's response
    if assistant_response:
        st.session_state["conversation_history"].append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Display the chat history dynamically
for message in st.session_state["conversation_history"]:
    with st.chat_message(message["role"]):  # Display user and assistant messages
        st.markdown(message["content"])
    # Process assistant response
    #with st.spinner("Thinking..."):
    #    response = process_input(user_input)

    # Display assistant response
    #st.session_state["messages"].append({"role": "assistant", "content": response})
    #with st.chat_message("assistant"):
    #    st.markdown(response)