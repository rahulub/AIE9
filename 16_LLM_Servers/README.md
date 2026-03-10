<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 16: LLM Servers</h1>

| 📰 Session Sheet                                  | ⏺️ Recording                           | 🖼️ Slides                                   | 👨‍💻 Repo       | 📝 Homework                                              | 📁 Feedback                        |
| ------------------------------------------------- | -------------------------------------- | ------------------------------------------- | ------------- | -------------------------------------------------------- | ---------------------------------- |
| [Session 16: LLM Servers](https://www.notion.so/) | [Recording!](https://us02web.zoom.us/) | [Session 16 Slides](https://www.canva.com/) | You are here! | [Session 16 Assignment: LLM Servers](https://forms.gle/) | [AIE9 Feedback](https://forms.gle) |

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU'RE FINISHED YOUR ASSIGNMENT !!!⚠️**

# Build 🏗️

In today's assignment, we'll be creating Fireworks AI endpoints, and then building a RAG application.

- 🤝 Breakout Room #1
  - Set-up Open Source Endpoint (Instructions [here](./ENDPOINT_SETUP.md)) ((This process may take 15-20min.))
  - Test Endpoint and Embeddings with the `endpoint_slammer.ipynb` notebook.

- 🤝 Breakout Room #2
  - Use the Open Source Endpoints to build a RAG LangGraph application

# Ship 🚢

The completed notebook and your RAG app/notebook!

### Deliverables

- A short Loom of either:
  - the notebook and the RAG application you built for the Main Homework Assignment; or
  - the notebook you created for the Advanced Build

# Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped a RAG application powered by open-source endpoints! 🎉🤖

🔍 Three Key Takeaways:
1️⃣
2️⃣
3️⃣

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

# Submitting You Homework [OPTIONAL]

## Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:

1. Follow the instructions in `ENDPOINT_SETUP.md`
2. Replace both `model` values in `endpoint_slammer.ipynb` with the `gpt-oss` endpoint you created in Step 1
3. Run the code cells in `endpoint_slammer.ipynb`
4. Respond to the questions in the section below
5. Build a sample RAG
6. Record a Loom video reviewing what you have learned from this session

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU HAVE FINISHED YOUR ASSIGNMENT !!!⚠️**

## Questions

### ❓ Question #1:

What is the difference between serverless and dedicated endpoints?

#### ✅ Answer:

## Serverless Endpoints
- **No setup required** — You use the endpoint directly (e.g., `accounts/fireworks/models/gpt-oss-20b`) without deploying anything.
- **Auto-scaled** — The provider manages capacity behind the scenes; resources scale with demand.
- **Pay-per-use** — You're billed for actual usage (tokens/requests), not idle time.
- **Best for** — Light or variable traffic, quick experiments, getting started without deployment hassle.

## Dedicated Endpoints
- **Requires deployment** — You provision your own endpoint (e.g., via `firectl`) with explicit min/max replicas.
- **Reserved capacity** — Resources are reserved for you, so you have guaranteed throughput and typically lower latency.
- **Pay for uptime** — You're billed for the time the deployment runs (e.g., hourly), even when idle.
- **Best for** — Production workloads, consistent traffic, or when you need predictable performance.

## Quick Comparison

| Aspect | Serverless | Dedicated |
|--------|------------|-----------|
| Setup | None | Requires deployment |
| Capacity | Shared, auto-scaled | Reserved for you |
| Cost model | Pay per use | Pay for uptime |
| Best for | Variable/sporadic traffic | Consistent, high traffic |
| Management | Hands-off | Must start/stop yourself |

For your assignment, use **serverless** if you want to avoid setup and billing management; use **dedicated** when you need guaranteed capacity and are willing to manage deployment and costs.

### ❓ Question #2:

Why is it important to consider token throughput and latency when choosing an LLM for user-facing applications?

#### ✅ Answer:

## Latency (Time to First Token)
- **User experience** — Users expect quick responses. High latency (seconds per response) leads to frustration, perceived unresponsiveness, and higher abandonment rates.
- **Conversation flow** — Chat interfaces rely on back-and-forth interaction. Slow replies break the sense of a natural dialogue.
- **Trust** — Consistent, fast responses feel reliable; slow or erratic ones feel broken.

## Token Throughput (Tokens per Second)
- **Streaming UX** — Many interfaces stream tokens as they're generated. Low throughput means words appear slowly, which feels laggy even if time-to-first-token is decent.
- **Concurrent users** — More users mean more requests at once. Throughput defines how many requests a model can handle in parallel without slowdown.
- **Cost and efficiency** — Higher throughput finishes each request faster, which can reduce total compute time and cost at scale.

## Summary

| Factor | Why It Matters |
|--------|----------------|
| **Latency** | Determines how quickly the user sees *something* happening — crucial for perceived responsiveness. |
| **Throughput** | Determines how fast the full response appears, especially when streaming. |
| **Together** | Both affect the user experience for chat-style or streaming UIs. A model that’s strong on quality but weak on latency/throughput can still feel poor in production. |

For user-facing apps, you typically need low latency and enough throughput to support streaming and concurrent users without noticeable lag — and you may need dedicated endpoints instead of serverless if traffic and performance requirements demand it.


## Activity 1: RAGAS Evaluation with Cost Analysis

Use RAGAS to evaluate your open-source Fireworks AI powered RAG app against an OpenAI `gpt-4.1-mini` powered equivalent. Compare retrieval quality, answer faithfulness, and end-to-end accuracy across both providers.

### RAGAS Scores Summary

| Metric | OpenAI | Fireworks | Difference |
|--------|--------|-----------|------------|
| **Context precision** | 0.72 | 0.23 | +0.49 (OpenAI) |
| **Context recall** | 0.90 | 0.20 | +0.70 (OpenAI) |
| **Faithfulness** | 0.88 | 0.10 | +0.78 (OpenAI) |
| **Answer relevancy** | 0.90 | 0.20 | +0.70 (OpenAI) |
| **Answer correctness** | 0.88 | 0.43 | +0.45 (OpenAI) |

### Summary

- **OpenAI** performs better across all metrics.
- **Fireworks** has much lower scores and often answers "I don't know."

### Likely Causes for Fireworks Gap

1. **Retrieval** — Fireworks embeddings may retrieve less relevant chunks (e.g., references vs. main content).
2. **Chat model** — `gpt-oss-20b` tends to default to "I don't know" even when the context is useful.



Additionally, instrument both pipelines with **LangSmith** to capture token usage and cost per query. Use LangSmith's tracing and cost dashboards to compare the total cost of running each provider at scale. Include your evaluation results, cost breakdown, and analysis in your Loom video.

## Advanced Activity: Local Models

Swap out the Fireworks AI endpoints for **locally-running open-source models** using [Ollama](https://ollama.com/) or another local inference server of your choice. Run both your embedding model and your chat model locally, and rebuild the RAG pipeline on top of them.

- Compare quality and latency between the local setup and your Fireworks AI hosted endpoint.
- Reflect: what are the trade-offs of local models vs. managed endpoints in a production setting?

Include your findings and a demo in your Loom video.
