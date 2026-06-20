import dotenv from "dotenv";
import { Anthropic } from "@anthropic-ai/sdk";
import { StateGraph, START, Annotation } from "@langchain/langgraph";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

dotenv.config({ path: path.join(__dirname, "../.env") });

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const state = Annotation.Root({
  question: Annotation(),
  findings: Annotation(),
  loopsCount: Annotation({
    default: () => 0,
    reducer: (_, updated) => updated,
  }),
  answer: Annotation(),
});

async function research(state) {
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    messages: [{ role: "user", content: state.question }],
    max_tokens: 1000,
  });

  const text = (response.content ?? [])
    .filter((item) => item.type === "text")
    .map((item) => item.text)
    .join("\n");

  return {
    answer: text,
    loopsCount: state.loopsCount + 1,
  };
}

const graph = new StateGraph(state)
  .addNode("research", research)
  .addEdge(START, "research")
  .compile();

async function main() {
  const [, , question] = process.argv;

  if (!question) {
    console.error("Usage: node hello-claude.js <question>");
    return;
  }

  const result = await graph.invoke({ question });
  console.log(result.answer);
}

await main();
