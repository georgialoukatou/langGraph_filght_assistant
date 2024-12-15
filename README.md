# Airlines Customer Support Assistant

This project implements an AI-powered Customer Support Assistant for **Swiss Airlines**, using **LangChain**, **LangGraph**, and **OpenAI's GPT**. The assistant interacts with users to answer queries, manage flight bookings, and provide company policies. It leverages advanced language models and a state-based execution graph for reliable, context-aware support.

---

## Features

### Flight Management Tools:
- Search for flights.
- Fetch user flight information.
- Update tickets to new flights.
- Cancel flight tickets.

### Policy Assistance:
- Retrieve and explain Swiss Airlines' policies.

### Interactive State Management:
- Uses **LangGraph** for dynamic control flow with persistent memory and fallback mechanisms.

### Error Handling:
- Detects tool errors and provides corrective suggestions.

### Enhanced AI-Powered Interaction:
- Employs **OpenAI GPT** for generating accurate and contextually relevant responses.
- Expands query bounds for better search results.

---

## Technologies Used
- **Python**: Core programming language.
- **LangChain**: For LLM (Large Language Model) integration and tooling.
- **LangGraph**: For state-based execution and workflow management.
- **OpenAI GPT**: Language model for conversational AI.
- **SQLite**: Lightweight database for flight information storage.
- **Pandas**: For data manipulation and analysis.
- **Requests**: For fetching remote data (e.g., database files).

---

## How It Works

### Tool-Based Workflow:
- Tools like `fetch_user_flight_information` and `search_flights` are integrated into a **state graph**.
- The assistant calls these tools dynamically based on user queries.

### State Management:
- The system uses **LangGraph's StateGraph** to maintain conversation state.
- A memory saver persists the entire graph state for continuity.

### Fallbacks and Error Recovery:
- A custom fallback mechanism ensures graceful handling of tool errors.

### Persistent Database:
- A **SQLite database (`travel2.sqlite`)** is used to store flight-related data.
- A backup file ensures the database can reset to its original state for tutorials or testing.

---

## Example Conversation

Users can ask questions like:
- *"What is the policy if I want to get a refund?"*
- *"When exactly is my flight? Can I change it?"*
