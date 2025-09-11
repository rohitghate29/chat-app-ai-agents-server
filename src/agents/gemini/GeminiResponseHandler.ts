import { GoogleGenerativeAI, GenerativeModel, ChatSession } from "@google/generative-ai";
import type { Channel, Event, MessageResponse, StreamChat } from "stream-chat";

export class GeminiResponseHandler {
  private message_text = "";
  private chunk_counter = 0;
  private is_done = false;
  private last_update_time = 0;

  constructor(
    private readonly genAI: GoogleGenerativeAI,
    private readonly model: GenerativeModel,
    private readonly chat: ChatSession,
    private readonly chatClient: StreamChat,
    private readonly channel: Channel,
    private readonly message: MessageResponse,
    private readonly onDispose: () => void
  ) {
    this.chatClient.on("ai_indicator.stop", this.handleStopGenerating);
  }

  run = async (userMessage: string, instructions: string) => {
    const { cid, id: message_id } = this.message;
    let isCompleted = false;

    try {
      // Send the user message with instructions
      const fullPrompt = `${instructions}\n\nUser message: ${userMessage}`;
      
      await this.channel.sendEvent({
        type: "ai_indicator.update",
        ai_state: "AI_STATE_GENERATING",
        cid: cid,
        message_id: message_id,
      });

      const result = await this.chat.sendMessageStream(fullPrompt);
      
      for await (const chunk of result.stream) {
        if (this.is_done) {
          break;
        }

        const chunkText = chunk.text();
        if (chunkText) {
          this.message_text += chunkText;
          const now = Date.now();
          if (now - this.last_update_time > 1000) {
            this.chatClient.partialUpdateMessage(message_id, {
              set: { text: this.message_text },
            });
            this.last_update_time = now;
          }
          this.chunk_counter += 1;
        }
      }

      // Check if the response requires function calls
      const response = await result.response;
      const functionCalls = response.functionCalls();
      if (functionCalls && functionCalls.length > 0) {
        await this.channel.sendEvent({
          type: "ai_indicator.update",
          ai_state: "AI_STATE_EXTERNAL_SOURCES",
          cid: cid,
          message_id: message_id,
        });

        const functionResults = [];
        for (const functionCall of functionCalls) {
          if (functionCall.name === "web_search") {
            try {
              const args = functionCall.args as { query: string };
              const searchResult = await this.performWebSearch(args.query);
              functionResults.push({
                name: functionCall.name,
                response: searchResult,
              });
            } catch (e) {
              console.error(
                "Error performing web search",
                e
              );
              functionResults.push({
                name: functionCall.name,
                response: JSON.stringify({ error: "failed to call tool" }),
              });
            }
          }
        }

        // Send the function results back to the model
        if (functionResults.length > 0) {
          const functionResultsText = functionResults.map(result => 
            `Function: ${result.name}\nResult: ${result.response}`
          ).join('\n\n');
          const followUpResult = await this.chat.sendMessageStream(functionResultsText);
          
          for await (const chunk of followUpResult.stream) {
            if (this.is_done) {
              break;
            }

            const chunkText = chunk.text();
            if (chunkText) {
              this.message_text += chunkText;
              const now = Date.now();
              if (now - this.last_update_time > 1000) {
                this.chatClient.partialUpdateMessage(message_id, {
                  set: { text: this.message_text },
                });
                this.last_update_time = now;
              }
              this.chunk_counter += 1;
            }
          }
        }
      }

      // Final update with complete message
      this.chatClient.partialUpdateMessage(message_id, {
        set: { text: this.message_text },
      });

      await this.channel.sendEvent({
        type: "ai_indicator.clear",
        cid: cid,
        message_id: message_id,
      });

      isCompleted = true;
    } catch (error) {
      console.error("An error occurred during the run:", error);
      await this.handleError(error as Error);
    } finally {
      await this.dispose();
    }
  };

  dispose = async () => {
    if (this.is_done) {
      return;
    }
    this.is_done = true;
    this.chatClient.off("ai_indicator.stop", this.handleStopGenerating);
    this.onDispose();
  };

  private handleStopGenerating = async (event: Event) => {
    if (this.is_done || event.message_id !== this.message.id) {
      return;
    }

    console.log("Stop generating for message", this.message.id);
    
    await this.channel.sendEvent({
      type: "ai_indicator.clear",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.dispose();
  };

  private handleError = async (error: Error) => {
    if (this.is_done) {
      return;
    }
    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_ERROR",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: {
        text: error.message ?? "Error generating the message",
        message: error.toString(),
      },
    });
    await this.dispose();
  };

  private performWebSearch = async (query: string): Promise<string> => {
    const TAVILY_API_KEY = process.env.TAVILY_API_KEY;

    if (!TAVILY_API_KEY) {
      return JSON.stringify({
        error: "Web search is not available. API key not configured.",
      });
    }

    console.log(`Performing web search for: "${query}"`);

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TAVILY_API_KEY}`,
        },
        body: JSON.stringify({
          query: query,
          search_depth: "advanced",
          max_results: 5,
          include_answer: true,
          include_raw_content: false,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Tavily search failed for query "${query}":`, errorText);
        return JSON.stringify({
          error: `Search failed with status: ${response.status}`,
          details: errorText,
        });
      }

      const data = await response.json();
      console.log(`Tavily search successful for query "${query}"`);

      return JSON.stringify(data);
    } catch (error) {
      console.error(
        `An exception occurred during web search for "${query}":`,
        error
      );
      return JSON.stringify({
        error: "An exception occurred during the search.",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  };
}
