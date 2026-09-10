import type {
  AgentChatRuntime,
  AgentChatState,
  ChatMessage,
} from "./chatState";
import type { ChatStreamState } from "./chatStreamEvents";
import { dequeue } from "./messageQueue";
import type { QueuedMessage } from "./messageQueue";

interface ChatTurnFinalizationOptions {
  agentId: string;
  assistantMessageId: string;
  sessionId: string | null;
  streamState: ChatStreamState;
  runtime: AgentChatRuntime;
  dequeueLocal: boolean;
  updateMessages: (
    updater: (messages: ChatMessage[]) => ChatMessage[],
  ) => void;
  patchState: (patch: Partial<AgentChatState>) => void;
  loadMessages: (
    agentId: string,
    sessionId: string,
  ) => Promise<ChatMessage[] | null>;
  onTurnComplete?: (agentId: string) => void;
  sendQueuedMessage: (
    agentId: string,
    text: string,
    sessionId: string | null,
    displayedMessageId?: string,
  ) => Promise<void>;
  dequeueMessage?: (agentId: string) => QueuedMessage | null;
  schedule?: (callback: () => void, delayMs: number) => void;
  now?: () => number;
}

export async function finalizeChatTurn({
  agentId,
  assistantMessageId,
  sessionId,
  streamState,
  runtime,
  dequeueLocal,
  updateMessages,
  patchState,
  loadMessages,
  onTurnComplete,
  sendQueuedMessage,
  dequeueMessage = dequeue,
  schedule = (callback, delayMs) => { setTimeout(callback, delayMs); },
  now = Date.now,
}: ChatTurnFinalizationOptions): Promise<void> {
  runtime.turnId = null;
  const stoppedByUser = runtime.userStopped;
  if (runtime.assistantMessageId === assistantMessageId) {
    runtime.assistantMessageId = null;
  }
  patchState({ isStreaming: false });
  runtime.controller = null;
  runtime.userStopped = false;

  if (!streamState.doneReceived) {
    // The SSE connection can end while the backend is persisting its final
    // assistant/tool result. Reload so the UI converges to that durable result.
    if (!stoppedByUser && sessionId) {
      try {
        await loadMessages(agentId, sessionId);
      } catch {
        // Best-effort reload.
      }
    }
  } else {
    updateMessages((previous) => {
      const targetId = runtime.assistantMessageId || assistantMessageId;
      const index = previous.findIndex((message) => message.id === targetId);
      const last = index >= 0 ? previous[index] : null;
      if (
        index >= 0
        && last?.role === "assistant"
        && last.isStreaming
      ) {
        const updated = previous.slice();
        updated[index] = {
          ...last,
          isStreaming: false,
          finishedAt: now(),
        };
        return updated;
      }
      return previous;
    });
  }

  onTurnComplete?.(agentId);
  if (!dequeueLocal) return;

  try {
    const next = dequeueMessage(agentId);
    if (next) {
      schedule(() => {
        void sendQueuedMessage(
          agentId,
          next.text,
          runtime.sessionId,
          next.messageId,
        );
      }, 50);
    }
  } catch {
    // Ignore local queue failures.
  }
}
