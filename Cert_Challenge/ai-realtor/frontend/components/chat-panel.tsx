"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ArrowUp, User, Bot, RotateCcw, Loader2 } from "lucide-react"
import ReactMarkdown from "react-markdown"

function TypingIndicator() {
  return (
    <div className="flex items-center gap-2 py-1">
      <div className="flex gap-1.5">
        <span
          className="h-2 w-2 rounded-full bg-primary/60 animate-bounce"
          style={{ animationDelay: "0ms", animationDuration: "1s" }}
        />
        <span
          className="h-2 w-2 rounded-full bg-primary/60 animate-bounce"
          style={{ animationDelay: "150ms", animationDuration: "1s" }}
        />
        <span
          className="h-2 w-2 rounded-full bg-primary/60 animate-bounce"
          style={{ animationDelay: "300ms", animationDuration: "1s" }}
        />
      </div>
      <span className="text-xs text-muted-foreground animate-pulse">
        Analyzing...
      </span>
    </div>
  )
}

interface ChatPanelProps {
  context: {
    address: string
    filename: string
    priorities: string[]
  }
  onReset: () => void
}

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
}

export function ChatPanel({ context, onReset }: ChatPanelProps) {
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [threadId, setThreadId] = useState<string | null>(null)
  const threadIdRef = useRef<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  threadIdRef.current = threadId

  const priorities = Array.isArray(context.priorities) ? context.priorities : []
  const prioritiesList = priorities.length > 0 ? priorities.join(", ") : "General property condition"
  const neighborhoodPrefs = priorities.filter((p) =>
    /school|peaceful|walkability|safety|amenities|neighborhood|crime|park|grocery|transit|commute|location|area|community/i.test(p)
  )
  const hasSchoolPref = priorities.some((p) => /school/i.test(p))
  const schoolInstruction = hasSchoolPref
    ? ` For School Quality: call web_search with "${context.address} schools" and "${context.address} school ratings" to get elementary, middle, and high school details with ratings. Present: school name, grade levels, and rating for each.`
    : ""
  const webSearchReminder =
    neighborhoodPrefs.length > 0
      ? `\n\n⚠️ MANDATORY: User selected: ${neighborhoodPrefs.join(", ")}. You MUST call web_search — inspection report has NONE of this. Include address "${context.address}" in queries.${schoolInstruction}`
      : ""

  const hasSpecificCategories = priorities.some(
    (p) =>
      /foundation|roof|plumbing|electrical|hvac|structure|water damage|mold|safety|cosmetic|pest|appliance|window|door|flooring/i.test(
        p
      )
  )
  const reportAllRedFlags =
    !hasSpecificCategories
      ? "\n\nWhen no specific inspection categories are selected, report ALL red flags found in the inspection report."
      : ""

  const contextStr =
    `Property Address: ${context.address}\n` +
    `User preferences while buying this property: ${prioritiesList}\n` +
    `Inspection report "${context.filename}" has been indexed. Search for and identify red flags across: ` +
    "structural issues, roof, foundation, electrical, plumbing, HVAC, water damage, mold, safety hazards. " +
    "For each red flag: 1) Issue description, 2) Severity (🔴 Critical / 🟠 Major / 🟡 Minor), 3) Page number. " +
    "Always ORDER red flags by decreasing severity: Critical first, then Major, then Minor. " +
    "ALWAYS include findings relevant to the user's custom preferences in your answer. " +
    "Tailor your analysis to the user's stated preferences." +
    reportAllRedFlags +
    webSearchReminder

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isStreaming) return

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: text.trim(),
      }
      setMessages((prev) => [...prev, userMsg])
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", content: "" },
      ])
      setIsStreaming(true)
      setError(null)

      try {
        const body: { message: string; context?: string; thread_id?: string } = {
          message: text.trim(),
          context: contextStr,
        }
        const tid = threadIdRef.current
        if (tid) body.thread_id = tid

        const base =
          process.env.NEXT_PUBLIC_BACKEND_URL ||
          (typeof window !== "undefined" ? "http://localhost:8000" : "")
        const url = base ? `${base}/api/chat` : "/api/chat"
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        })

        if (!res.ok) throw new Error(`Server error: ${res.status}`)

        const newThreadId = res.headers.get("X-Thread-Id")
        if (newThreadId && !tid) {
          threadIdRef.current = newThreadId
          setThreadId(newThreadId)
        }

        const reader = res.body?.getReader()
        if (!reader) throw new Error("No response body")

        const decoder = new TextDecoder()
        let accumulated = ""

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const chunk = decoder.decode(value, { stream: true })
          accumulated += chunk
          setMessages((prev) => {
            const updated = [...prev]
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: accumulated,
            }
            return updated
          })
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Request failed")
        setMessages((prev) => {
          const updated = [...prev]
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: `Error: ${err instanceof Error ? err.message : "Request failed"}`,
          }
          return updated
        })
      } finally {
        setIsStreaming(false)
      }
    },
    [contextStr]
  )

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    const text = input.trim()
    if (!text || isStreaming) return
    setInput("")
    sendMessage(text)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleClearChat = async () => {
    if (threadId) {
      try {
        const base =
          process.env.NEXT_PUBLIC_BACKEND_URL ||
          (typeof window !== "undefined" ? "http://localhost:8000" : "")
        const url = base ? `${base}/api/chat/thread/${threadId}` : `/api/chat/thread/${threadId}`
        await fetch(url, { method: "DELETE" })
      } catch (_) {}
    }
    setThreadId(null)
    setMessages([])
    setError(null)
    onReset()
  }

  // Auto-send initial analysis request (once)
  const hasSentInitial = useRef(false)
  useEffect(() => {
    if (!hasSentInitial.current && context.filename) {
      hasSentInitial.current = true
      sendMessage(
        "Please analyze this property inspection report and provide a comprehensive assessment based on my priorities."
      )
    }
  }, [context.filename, sendMessage])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div
        className={`flex items-center justify-between border-b border-border px-5 py-3 bg-card transition-colors duration-300 ${
          isStreaming ? "bg-primary/5" : ""
        }`}
      >
        <div className="flex flex-col">
          <h2 className="text-sm font-semibold text-foreground">Analysis Chat</h2>
          <p className="text-xs text-muted-foreground truncate max-w-[300px]">
            {context.address}
          </p>
        </div>
        <Button
          onClick={handleClearChat}
          variant="ghost"
          size="sm"
          className="text-xs text-muted-foreground hover:text-foreground gap-1"
        >
          <RotateCcw className="h-3 w-3" />
          New Analysis
        </Button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4 relative" ref={scrollRef}>
        {isStreaming && (
          <div className="absolute top-0 left-0 right-0 h-0.5 bg-primary/15 overflow-hidden z-10">
            <div
              className="h-full w-1/4 bg-primary/50 rounded-full"
              style={{ animation: "progress-shimmer 1.8s ease-in-out infinite" }}
            />
          </div>
        )}
        <div className="flex flex-col gap-6 max-w-none">
          {messages.map((message) => (
            <div key={message.id} className="flex gap-3">
              <div
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${
                  message.role === "assistant"
                    ? "bg-accent text-accent-foreground"
                    : "bg-secondary text-secondary-foreground"
                }`}
              >
                {message.role === "assistant" ? (
                  <Bot className="h-4 w-4" />
                ) : (
                  <User className="h-4 w-4" />
                )}
              </div>
              <div className="flex-1 min-w-0 pt-0.5">
                <p className="text-xs font-medium text-muted-foreground mb-1.5">
                  {message.role === "assistant" ? "AI Realtor" : "You"}
                </p>
                <div className="text-sm text-foreground [&_h1]:text-lg [&_h1]:font-bold [&_h1]:mb-2 [&_h1]:mt-3 [&_h2]:text-base [&_h2]:font-semibold [&_h2]:mb-2 [&_h2]:mt-3 [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mb-1 [&_h3]:mt-2 [&_p]:mb-2 [&_p]:leading-relaxed [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:mb-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:mb-2 [&_li]:mb-1 [&_li]:leading-relaxed [&_strong]:font-semibold [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs">
                  {message.role === "assistant" ? (
                    message.content ? (
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    ) : isStreaming && message.id === messages[messages.length - 1]?.id ? (
                      <TypingIndicator />
                    ) : null
                  ) : (
                    message.content
                  )}
                </div>
              </div>
            </div>
          ))}

        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="mx-4 mb-2 rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {/* Follow-up chat — fixed at bottom, always visible */}
      <div
        className={`border-t border-border p-4 bg-card shrink-0 min-h-[120px] transition-all duration-300 ${
          isStreaming ? "ring-2 ring-primary/20 ring-inset" : ""
        }`}
      >
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-medium text-foreground">
            Follow-up questions
          </p>
          {isStreaming && (
            <span className="flex items-center gap-1.5 text-xs text-primary animate-pulse">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Processing...
            </span>
          )}
        </div>
        {messages.some((m) => m.role === "assistant" && m.content.length > 0) &&
          !isStreaming && (
            <p className="text-xs text-muted-foreground mb-2">
              Ask about specific findings, repair costs, schools, or the report.
            </p>
          )}
        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleSubmit()
          }}
          className="flex gap-2"
        >
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your question... (e.g. What about the roof? Estimated repair costs? Schools in the area?)"
            disabled={isStreaming}
            rows={1}
            className="min-h-[48px] max-h-[120px] resize-none bg-background border-border leading-relaxed"
            aria-label="Follow-up question"
          />
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isStreaming}
            className="h-11 w-11 shrink-0 bg-primary text-primary-foreground hover:bg-primary/90"
            aria-label="Send message"
          >
            <ArrowUp className="h-4 w-4" />
          </Button>
        </form>
        <p className="text-[10px] text-muted-foreground mt-2 text-center">
          AI analysis is for informational purposes only. Consult a professional
          inspector for official assessments.
        </p>
      </div>
    </div>
  )
}
