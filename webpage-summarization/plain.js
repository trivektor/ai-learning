import { fetchWebpageContent, summarize, reviewSummary } from "./langgraph.js";

async function main() {
  const [, , url] = process.argv;

  if (!url) {
    console.error("Usage: node webpage-summarization.js <url>");
    return;
  }

  let state = {
    url,
    content: "",
    summary: "",
    feedback: "",
    reviewCount: 0,
    score: 0,
  };
  state = { ...state, ...(await fetchWebpageContent(state)) };
  while (true) {
    state = { ...state, ...(await summarize(state)) };
    state = { ...state, ...(await reviewSummary(state)) };
    if (state.score >= 80) break;
    if (state.reviewCount > 3) break;
  }
  console.log(state.summary);
}

main();
