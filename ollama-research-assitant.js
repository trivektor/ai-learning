import { ChatOllama } from "@langchain/ollama";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { TavilySearch } from "@langchain/tavily";
import dotenv from "dotenv";

dotenv.config();

const ollama = new ChatOllama({
  model: "gemma4:e2b",
});

const search = new TavilySearch({
  maxResults: 3,
  apiKey: process.env.TAVILY_API_KEY,
});

const GraphState = Annotation.Root({
  question: Annotation(),
  findings: Annotation(),
  loopsCount: Annotation({
    default: () => 0,
    reducer: (_, updated) => updated,
  }),
  answer: Annotation(),
});

async function research(state) {
  const { results } = (await search.invoke({
    query: state.question,
  })) || { results: [] };

  //console.log(JSON.stringify({ state, researchResults: results }, null, 2));

  return {
    findings: results,
    loopsCount: state.loopsCount + 1,
  };
}

async function decide(state) {
  const prompt = `
		Question: ${state.question}
		Research findings:
		
		${state.findings.map((result, index) => `${index + 1}. ${result.title}\n\n${result.content}`).join("\n\n\n")}
		
		If you have enough info, write the final answer. Otherwise reply with exactly: NEED_MORE_RESEARCH
	`.trim();

  //console.log(prompt);

  const response = await ollama.invoke(prompt);

  if (response.content.includes("NEED_MORE_RESEARCH")) {
    return {};
  }

  return {
    answer: response.content,
  };
}

function route(state) {
  if (state.answer) return END;
  if (state.loopsCount > 3) return END;
  return "research";
}

const graph = new StateGraph(GraphState)
  .addNode("research", research)
  .addNode("decide", decide)
  .addEdge(START, "research")
  .addEdge("research", "decide")
  .addConditionalEdges("decide", route)
  .compile();

async function main() {
  const [, , question] = process.argv;

  if (!question) {
    console.error("Usage: node ollama-research-assitant.js <question>");
    return;
  }
  const result = await graph.invoke({ question });

  console.log(result.answer);
}

await main();
