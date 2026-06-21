import { StateSchema, StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod/v4";
import dotenv from "dotenv";
import * as cheerio from "cheerio";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

dotenv.config();

const openai = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4o-mini",
});

const structured = openai.withStructuredOutput(
  z.object({
    score: z.number().min(0).max(100),
    feedback: z.string(),
  }),
);

async function fetchWebpageContent(state) {
  const response = await fetch(state.url);
  const html = await response.text();

  return { content: cheerio.load(html).text() };
}

async function summarize(state) {
  const feedbackSection = state.feedback
    ? `\nPrevious attempt feedback — address this:\n${state.feedback}\n`
    : "";
  const message = `
  Below is the content of a webpage. Summarize the article in bullet points.
  ${state.content}
  ${feedbackSection}
  `;
  const response = await openai.invoke([new HumanMessage(message)]);

  return { summary: response.text };
}

async function reviewSummary(state) {
  const message = `
  Below is the summary of an article at ${state.url}. Review the summary and provide a score between 0 and 100 and provide feedback.
  ${state.summary}
  `;
  const response = await structured.invoke([new HumanMessage(message)]);

  return {
    score: response.score,
    feedback: response.feedback,
    reviewCount: state.reviewCount + 1,
  };
}

function route(state) {
  if (state.score >= 80) return END;
  if (state.reviewCount > 3) return END;
  return "summarize";
}

const stateSchema = new StateSchema({
  url: z.string(),
  content: z.string(),
  summary: z.string(),
  feedback: z.string().default(""),
  reviewCount: z.number().default(0),
  score: z.number().default(0),
});

const graph = new StateGraph(stateSchema)
  .addNode("fetchWebpageContent", fetchWebpageContent)
  .addNode("summarize", summarize)
  .addNode("reviewSummary", reviewSummary)
  .addEdge(START, "fetchWebpageContent")
  .addEdge("fetchWebpageContent", "summarize")
  .addEdge("summarize", "reviewSummary")
  .addConditionalEdges("reviewSummary", route)
  .compile();

async function main() {
  const [, , url] = process.argv;

  if (!url) {
    console.error("Usage: node webpage-summarization.js <url>");
    return;
  }

  const result = await graph.invoke({ url });
  console.log(result.summary);
}

await main();
