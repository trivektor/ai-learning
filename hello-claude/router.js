import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { Anthropic } from "@anthropic-ai/sdk";

const MODEL = "claude-haiku-4-5-20251001";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

dotenv.config({ path: path.join(__dirname, "../.env") });

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const state = Annotation.Root({
  question: Annotation(),
  route: Annotation(),
  answer: Annotation(),
  loopsCount: Annotation({
    default: () => 0,
    reducer: (_, updated) => updated,
  }),
});

async function askClaude(options) {
  const response = await anthropic.messages.create({
    model: options.model ?? MODEL,
    messages: options.messages,
    system: options.system,
    max_tokens: options.max_tokens ?? 1000,
    output_config: options.output_config,
  });

  const textBlock = response.content.find((item) => item.type === "text");

  if (!textBlock) {
    throw new Error("No text block found in response");
  }

  const text = textBlock.text;

  return text;
}

async function classify(state) {
  const systemPrompt = `
  Classify the user's prompt into exactly one category.

Categories:

- code:
  Programming, software engineering, debugging, code review,
  architecture, databases, frameworks, APIs, or developer tools.

- explain:
  A request to teach or explain a non-programming concept.

- general:
  Anything that does not fit the other categories.

Programming explanations belong in "code".
Examples:
- "Explain closures" → code
- "Fix this TypeScript error" → code
- "Explain photosynthesis" → explain
- "Write a birthday message" → general
`.trim();

  const response = await askClaude({
    model: "claude-haiku-4-5-20251001",
    messages: [{ role: "user", content: state.question }],
    system: systemPrompt,
    max_tokens: 1000,
    output_config: {
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            route: {
              type: "string",
              enum: ["code", "explain", "general"],
            },
          },
          required: ["route"],
          additionalProperties: false,
        },
      },
    },
  });
  const json = JSON.parse(response);

  return { route: json.route };
}

function route(state) {
  return state.route;
}

async function softwareEngineer(state) {
  const systemPrompt = `You are an experienced software engineer. Answer the question below.
Give technically precise answers and include code when useful.

Question: ${state.question}
`;

  const answer = await askClaude({
    messages: [{ role: "user", content: state.question }],
    system: systemPrompt,
  });

  return { answer };
}

async function generalAssistant(state) {
  const systemPrompt = `You are a general assistant. Answer the question below.
Give a concise answer and include examples when useful.

Question: ${state.question}
`;

  const answer = await askClaude({
    messages: [{ role: "user", content: state.question }],
    system: systemPrompt,
  });

  return { answer };
}

async function teacher(state) {
  const systemPrompt = `You are a teacher. Answer the question below.
Give a concise answer and include examples when useful.

Question: ${state.question}
`;

  const answer = await askClaude({
    messages: [{ role: "user", content: state.question }],
    system: systemPrompt,
  });

  return { answer };
}

const graph = new StateGraph(state)
  .addNode("classify", classify)
  .addNode("softwareEngineer", softwareEngineer)
  .addNode("generalAssistant", generalAssistant)
  .addNode("teacher", teacher)
  .addEdge(START, "classify")
  .addConditionalEdges("classify", route, {
    code: "softwareEngineer",
    explain: "teacher",
    general: "generalAssistant",
  })
  .addEdge("softwareEngineer", END)
  .addEdge("teacher", END)
  .addEdge("generalAssistant", END)
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
