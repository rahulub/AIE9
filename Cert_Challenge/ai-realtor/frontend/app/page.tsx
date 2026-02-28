"use client"

import { useState } from "react"
import { InspectionForm } from "@/components/inspection-form"
import { ChatPanel } from "@/components/chat-panel"
import { Building2, Shield, MessageCircle, Upload } from "lucide-react"

interface FormData {
  address: string
  filename: string
  priorities: string[]
}

export default function Page() {
  const [formData, setFormData] = useState<FormData | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleFormSubmit = (data: FormData) => {
    setIsSubmitting(true)
    setFormData(data)
  }

  const handleReset = () => {
    setFormData(null)
    setIsSubmitting(false)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="mx-auto max-w-6xl flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Building2 className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-serif text-xl text-foreground">AI Realtor</span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Shield className="h-3.5 w-3.5" />
            <span>AI-Powered Analysis</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      {!formData ? (
        <main className="mx-auto max-w-2xl px-6 py-12">
          {/* Hero */}
          <div className="mb-10 text-center">
            <h1 className="font-serif text-3xl md:text-4xl text-foreground mb-3 text-balance">
              Property Analyzer
            </h1>
            <p className="text-muted-foreground leading-relaxed max-w-lg mx-auto text-pretty">
              Get AI-powered insights on your property inspection report. Understand key findings, prioritized by what matters most to you.
            </p>
          </div>

          {/* How it works */}
          <div className="grid grid-cols-3 gap-4 mb-10">
            {[
              { icon: Building2, label: "Enter Property", desc: "Provide the address" },
              { icon: Upload, label: "Upload Report", desc: "PDF or paste text" },
              { icon: MessageCircle, label: "Get Analysis", desc: "Chat with your results" },
            ].map((step, i) => (
              <div
                key={step.label}
                className="flex flex-col items-center gap-2 rounded-lg border border-border bg-card p-4 text-center"
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-md bg-secondary">
                  <step.icon className="h-4 w-4 text-secondary-foreground" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-foreground">{step.label}</p>
                  <p className="text-[11px] text-muted-foreground">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Form card */}
          <div className="rounded-xl border border-border bg-card p-6 md:p-8 shadow-sm">
            <InspectionForm onSubmit={handleFormSubmit} isLoading={isSubmitting} />
          </div>
        </main>
      ) : (
        <main className="mx-auto max-w-4xl px-4 pb-4 flex flex-col min-h-[calc(100vh-65px)]">
          <div className="flex-1 min-h-0 rounded-none md:rounded-xl md:my-6 border border-border bg-card shadow-sm overflow-hidden flex flex-col">
            <ChatPanel context={formData} onReset={handleReset} />
          </div>
        </main>
      )}
    </div>
  )
}
