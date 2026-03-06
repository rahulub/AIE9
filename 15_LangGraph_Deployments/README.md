<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 15: Build & Serve Agentic Graphs with LangGraph</h1>

| 📰 Session Sheet                                             | ⏺️ Recording                           | 🖼️ Slides                                  | 👨‍💻 Repo    | 📝 Homework                                      | 📁 Feedback                                          |
| ------------------------------------------------------------ | -------------------------------------- | ------------------------------------------- | ------------- | ------------------------------------------------ | ---------------------------------------------------- |
| [Agent Servers](https://github.com/AI-Maker-Space/AIE9/tree/main/00_Docs/Session_Sheets/15_Agent_Servers) |[Recording!](https://us02web.zoom.us/rec/share/lORjByDju6fv4TdE3r93dorY3aNgmSKL_Qk_cX_AMcCQ6cNfSW77unaA1LMVV60.OcI8uEnfVmRAgjSn) <br> passcode: `Dc@&pv1T`| [Session 15 Slides](https://www.canva.com/design/DAG-EJqkRaM/FR3WG_yMA5_BqbWpQlHR9g/edit?utm_content=DAG-EJqkRaM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 15 Assignment: Agent Servers](https://forms.gle/Vb3HNDsyVPQ1jqKX7) | [Feedback 3/3](https://forms.gle/kYmhbVUEMog16mKv8) |

### Prerequisites

Before starting, ensure you have the following:

- **Python 3.11+** installed
- An **OpenAI API Key**
- A **Tavily API Key**
- (Optional) **LangSmith** credentials for tracing

Create a `.env` file in this directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
2. Run `uv sync` to install dependencies.

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API from a different terminal
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangSmith Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

>NOTE: This can be done in place of the Main Assignment

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.

When submitting, provide:
- Your Loom video link demonstrating the MCP server integration
- The GitHub URL to your completed Advanced Build

Have fun!
</details>

### Questions & Activities

#### Question 1:
What is the key architectural difference between the `simple_agent` and `agent_with_helpfulness` graphs? Specifically, explain how the helpfulness evaluation loop works and what mechanisms are in place to prevent it from running indefinitely.

##### Answer:

## Key Architectural Difference: `simple_agent` vs `agent_with_helpfulness`

## Overview

**`simple_agent`** is a single loop: model → (maybe tools) → END. After the model responds, `tools_condition` decides whether to run tools or END.

**`agent_with_helpfulness`** adds a **helpfulness evaluation node** before ending. After the model responds (without tool calls), it routes to a helpfulness evaluator that decides whether to end or continue the loop.

## How the Helpfulness Evaluation Loop Works

1. **Routing after the agent** (`route_to_action_or_helpfulness`): If the last message has tool calls → go to `action`. Otherwise → go to `helpfulness`.

2. **Helpfulness evaluation** (`helpfulness_node`): Compares the initial user query (first message) with the latest response using a structured output (`HelpfulnessResult`). It injects a synthetic message `"HELPFULNESS:Y"` or `"HELPFULNESS:N"` into the state.

3. **Decision and loop control** (`helpfulness_decision`): Reads that synthetic message:
   - `"HELPFULNESS:Y"` → END
   - `"HELPFULNESS:N"` → route back to `agent`
   - `"HELPFULNESS:END"` → END (safety exit)

### Safeguards Against Infinite Loops

**Message-count guard** (`helpfulness_node`, lines 56–57):

if len(state["messages"]) > 10:
    return {"messages": [AIMessage(content="HELPFULNESS:END")]}

Once there are more than 10 messages, the helpfulness node injects "HELPFULNESS:END" instead of running the evaluator, which causes helpfulness_decision to terminate the graph and prevents unbounded iterations.


#### Question 2:
What is the role of `langgraph.json` in the LangGraph Deployments? Describe each of its key fields and how the platform uses this file to discover and serve your graphs.

##### Answer:

## The Role of `langgraph.json` in LangGraph Deployments

`langgraph.json` is the main configuration file for the LangGraph CLI and platform. It tells the system how to discover, load, and serve your graphs. The LangGraph CLI uses it as the single entry point for `langgraph dev`, `langgraph build`, and `langgraph up`.

## Key Fields

| Field | Purpose |
|-------|---------|
| **`version`** | Schema version of the configuration (e.g., `1`). |
| **`dependencies`** | **Required.** Where to find Python packages (`pyproject.toml`, `setup.py`, `requirements.txt`). Use `"."` for the current directory or a path like `"./local_package"` so the server installs dependencies correctly. |
| **`graphs`** | **Required.** Maps graph IDs to their compiled graph locations. Format: `"graph_id": "module.path:variable_or_function"`. Each entry points to a `CompiledStateGraph` or a function that returns one. The platform dynamically imports these at startup. |
| **`assistants`** | Optional. Defines assistant presets for the API and Studio. Each assistant has `graph_id` (which graph it runs), plus `name` and `description` for the UI. Lets you expose multiple "assistants" (e.g., `agent`, `agent_helpful`) backed by different graphs. |
| **`env`** | Path to a `.env` file or mapping of environment variables. Ensures API keys and other secrets are loaded when the server starts. |
| **`python_version`** | Python version for builds and runtime (e.g., `"3.13"`). |

## How the Platform Uses This File

1. **Discovery**: At startup (e.g., `langgraph dev`), the CLI uses the `graphs` block to find each graph. It dynamically imports each `module.path:variable` (e.g., `app.graphs.simple_agent:graph`). The `collect_graphs_from_env` step you saw in the error trace is this discovery phase.

2. **Graph loading**: For each graph ID, the runtime imports the module and retrieves the specified symbol. If something fails during import (e.g., a missing `TAVILY_API_KEY`), the graph fails to load with a `GraphLoadError`.

3. **Assistants**: The `assistants` block seeds the API and Studio with named assistants. When you invoke an assistant ID (e.g., `agent`, `agent_helpful`), the server uses the associated `graph_id` to select which graph to run.

4. **API surface**: Using these graphs and assistants, the Agent Server exposes endpoints for runs, threads, assistants, etc., all driven by this configuration.


#### Activity #1:
Create your own agent graph! Build a new graph in `app/graphs/` with a custom evaluation node (e.g., a vibe checker, a fact verifier, a summarizer — get creative!). Register it in `langgraph.json`, serve it with `uv run langgraph dev`

##### Answer:



# Ship 🚢

- The completed notebook.
- 5min. Loom Video

# Share 🚀

- Walk through your notebook and explain what you've completed in the Loom video
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

# Submitting Your Homework

### Main Homework Assignment

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your AIE9 repo:
    - _(You should have completed this process already.)_ For your initial repo setup, see [Initial_Setup](https://github.com/AI-Maker-Space/AIE9/tree/main/00_Docs/Prerequisites/Initial_Setup)
    - To get the latest updates from AI Makerspace into your own AIE9 repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `15_LangGraph_Platform` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Answer Questions 1 - 2 using the `##### Answer:` markdown cell below them in the README
4. Complete Activity #1 in the README
5. Add, commit and push your modified files to your GitHub repository.

When submitting your homework, provide:
- Your Loom video link
- The GitHub URL to the `15_LangGraph_Platform` folder on your assignment branch
