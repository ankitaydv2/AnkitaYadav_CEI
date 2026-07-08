# 🤖 Week 8 - Single Agentic AI Pipeline

## 📌 Project Overview

This project implements a **Single-Agent Smart Assistant** that understands user queries, identifies the user's intent, routes the request to the appropriate tool, and returns the response in a structured JSON format.

The project demonstrates the fundamentals of **Agentic AI**, where an agent acts as a decision-maker by selecting the correct tool based on the user's input.

---

## 🎯 Objectives

- Build a Single-Agent AI Pipeline
- Implement intent-based routing
- Integrate multiple tools
- Return structured JSON responses
- Handle invalid or unsupported queries gracefully

---

## 🛠️ Tools Implemented

### 1️⃣ Calculator Tool
Performs basic mathematical calculations.

**Example**
```
Input:
Calculate 20 + 5

Output:
25
```

---

### 2️⃣ Keyword Extraction Tool
Extracts important keywords from the given text.

**Example**
```
Input:
Extract keywords from Artificial Intelligence is transforming industries

Output:
["artificial", "intelligence", "transforming", "industries"]
```

---

### 3️⃣ Time Tool (Bonus Feature)
Returns the current system time.

**Example**
```
Input:
What is the time?

Output:
06:45 PM
```

---

## 🤖 Agent Workflow

```
                User Query
                     │
                     ▼
             Intent Detection
                     │
      ┌──────────────┼──────────────┐
      │              │              │
 Calculator     Keyword Tool     Time Tool
      │              │              │
      └──────────────┼──────────────┘
                     │
                     ▼
              JSON Response
```

---

## 📂 Project Features

- Intent-based query routing
- Calculator tool integration
- Keyword extraction
- Current time retrieval
- Interactive user mode
- Structured JSON output
- Basic error handling

---

## 🧪 Sample Output

### Calculator

```json
{
    "type": "calculation",
    "result": "25"
}
```

### Keyword Extraction

```json
{
    "type": "keywords",
    "result": [
        "artificial",
        "intelligence",
        "transforming",
        "industries"
    ]
}
```

### Time

```json
{
    "type": "time",
    "result": "06:45 PM"
}
```

### General Query

```json
{
    "type": "general",
    "result": "I'm a simple AI assistant. I can perform calculations, extract keywords, and tell the current time."
}
```

---

## 💻 Technologies Used

- Python
- Regular Expressions (`re`)
- JSON
- Datetime Module

---

## 📖 Learning Outcomes

Through this project, I learned:

- Fundamentals of Agentic AI
- Intent detection and routing
- Tool integration
- Structured JSON responses
- Building a simple AI agent pipeline
- Basic error handling in Python

---

## 🚀 Future Enhancements

- Weather API integration
- Wikipedia Search Tool
- Sentiment Analysis
- Unit Converter
- Logging and execution history
- LLM integration using Gemini or OpenAI

---

## 👩‍💻 Author

**Ankita Yadav**

B.Tech Computer Science Engineering  
Jaipur Engineering College and Research Centre (JECRC)
