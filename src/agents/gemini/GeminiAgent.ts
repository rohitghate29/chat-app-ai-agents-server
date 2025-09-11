import { GoogleGenerativeAI, GenerativeModel, SchemaType } from "@google/generative-ai";
import type { Channel, DefaultGenerics, Event, StreamChat } from "stream-chat";
import type { AIAgent } from "../types";
import { GeminiResponseHandler } from "./GeminiResponseHandler";

export class GeminiAgent implements AIAgent {
  private genAI?: GoogleGenerativeAI;
  private model?: GenerativeModel;
  private lastInteractionTs = Date.now();

  private handlers: GeminiResponseHandler[] = [];

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel
  ) {}

  dispose = async () => {
    this.chatClient.off("message.new", this.handleMessage);
    await this.chatClient.disconnectUser();

    this.handlers.forEach((handler) => handler.dispose());
    this.handlers = [];
  };

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = (): number => this.lastInteractionTs;

  init = async () => {
    const apiKey = process.env.GEMINI_API_KEY as string | undefined;
    if (!apiKey) {
      throw new Error("Gemini API key is required");
    }

    this.genAI = new GoogleGenerativeAI(apiKey);
    this.model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash",
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 1000,
      },
      tools: [
        {
          functionDeclarations: [
            {
              name: "web_search",
              description: "Search the web for current information, news, facts, or research on any topic",
              parameters: {
                type: SchemaType.OBJECT,
                properties: {
                  query: {
                    type: SchemaType.STRING,
                    description: "The search query to find information about",
                  },
                },
                required: ["query"],
              },
            },
          ],
        },
      ],
    });

    this.chatClient.on("message.new", this.handleMessage);
  };

  private getWritingAssistantPrompt = (context?: string): string => {
    const currentDate = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    return `You are an expert AI Writing Assistant. Your primary purpose is to be a collaborative writing partner.

**Your Core Capabilities:**
- Content Creation, Improvement, Style Adaptation, Brainstorming, and Writing Coaching.
- **Web Search**: You have the ability to search the web for up-to-date information using the 'web_search' tool.
- **Current Date**: Today's date is ${currentDate}. Please use this for any time-sensitive queries.

**Crucial Instructions:**
1.  **ALWAYS use the 'web_search' tool when the user asks for current information, news, or facts.** Your internal knowledge is outdated.
2.  When you use the 'web_search' tool, you will receive a JSON object with search results. **You MUST base your response on the information provided in that search result.** Do not rely on your pre-existing knowledge for topics that require current information.
3.  Synthesize the information from the web search to provide a comprehensive and accurate answer. Cite sources if the results include URLs.

**Response Format:**
- Be direct and production-ready.
- Use clear formatting.
- Never begin responses with phrases like "Here's the edit:", "Here are the changes:", or similar introductory statements.
- Provide responses directly and professionally without unnecessary preambles.

**Writing Context**: ${context || "General writing assistance."}

Your goal is to provide accurate, current, and helpful written content. Failure to use web search for recent topics will result in an incorrect answer.`;
  };

  private handleMessage = async (e: Event<DefaultGenerics>) => {
    if (!this.genAI || !this.model) {
      console.log("Gemini not initialized");
      return;
    }

    if (!e.message || e.message.ai_generated) {
      return;
    }

    const message = e.message.text;
    if (!message) return;

    this.lastInteractionTs = Date.now();

    const writingTask = (e.message.custom as { writingTask?: string })
      ?.writingTask;
    const context = writingTask ? `Writing Task: ${writingTask}` : undefined;
    const instructions = this.getWritingAssistantPrompt(context);

    const { message: channelMessage } = await this.channel.sendMessage({
      text: "",
      ai_generated: true,
    });

    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_THINKING",
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });

    const chat = this.model.startChat({
      history: [],
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 8192,
      },
    });

    const handler = new GeminiResponseHandler(
      this.genAI,
      this.model,
      chat,
      this.chatClient,
      this.channel,
      channelMessage,
      () => this.removeHandler(handler)
    );
    this.handlers.push(handler);
    void handler.run(message, instructions);
  };

  private removeHandler = (handlerToRemove: GeminiResponseHandler) => {
    this.handlers = this.handlers.filter(
      (handler) => handler !== handlerToRemove
    );
  };
}
