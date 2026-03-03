"use client"

import { useState, useRef } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import {
  MapPin,
  ArrowRight,
  X,
  Upload,
  File,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react"

const PRIORITY_CATEGORIES = [
  {
    label: "Home Structure",
    items: [
      "Structural Integrity",
      "Foundation",
      "Roof Condition",
      "Water Damage",
      "Mold",
    ],
  },
  {
    label: "Home Systems",
    items: [
      "Plumbing",
      "Electrical",
      "HVAC",
      "Energy Efficiency",
      "Safety Hazards",
    ],
  },
  {
    label: "Home Details",
    items: [
      "Cosmetic Issues",
      "Pest/Termite",
      "Appliances",
      "Windows & Doors",
      "Flooring",
    ],
  },
  {
    label: "Neighborhood",
    items: [
      "School Quality",
      "Peaceful Neighborhood",
      "Walkability",
      "Safety & Crime",
      "Nearby Amenities",
    ],
  },
]

interface InspectionFormProps {
  onSubmit: (data: {
    address: string
    filename: string
    priorities: string[]
  }) => void
  isLoading: boolean
}

export function InspectionForm({ onSubmit, isLoading }: InspectionFormProps) {
  const [address, setAddress] = useState("")
  const [priorities, setPriorities] = useState<string[]>([])
  const [customPriority, setCustomPriority] = useState("")

  // PDF upload state — uses backend /api/ingest (RAG)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [ingestedFilename, setIngestedFilename] = useState("")
  const [uploadStatus, setUploadStatus] = useState<
    "idle" | "uploading" | "success" | "error"
  >("idle")
  const [uploadError, setUploadError] = useState("")
  const [pdfPages, setPdfPages] = useState(0)
  const [chunksIngested, setChunksIngested] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const togglePriority = (priority: string) => {
    setPriorities((prev) =>
      prev.includes(priority)
        ? prev.filter((p) => p !== priority)
        : prev.length < 10
          ? [...prev, priority]
          : prev
    )
  }

  const addCustomPriority = () => {
    const trimmed = customPriority.trim()
    if (trimmed && !priorities.includes(trimmed) && priorities.length < 10) {
      setPriorities((prev) => [...prev, trimmed])
      setCustomPriority("")
    }
  }

  const removePriority = (priority: string) => {
    setPriorities((prev) => prev.filter((p) => p !== priority))
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (file.type !== "application/pdf") {
      setUploadStatus("error")
      setUploadError("Please select a PDF file.")
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setUploadStatus("error")
      setUploadError("File size must be under 10MB.")
      return
    }

    setUploadedFile(file)
    setUploadStatus("uploading")
    setUploadError("")

    try {
      const formData = new FormData()
      formData.append("file", file)

      const res = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      })

      const data = await res.json()

      if (!res.ok) {
        setUploadStatus("error")
        setUploadError(data.detail || data.error || "Failed to ingest PDF.")
        return
      }

      setIngestedFilename(data.filename || file.name)
      setPdfPages(data.pages || 0)
      setChunksIngested(data.chunks_ingested || 0)
      setUploadStatus("success")
    } catch {
      setUploadStatus("error")
      setUploadError("Failed to upload file. Please try again.")
    }
  }

  const handleRemoveFile = () => {
    setUploadedFile(null)
    setIngestedFilename("")
    setUploadStatus("idle")
    setUploadError("")
    setPdfPages(0)
    setChunksIngested(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!address.trim() || !ingestedFilename || priorities.length === 0)
      return
    onSubmit({ address, filename: ingestedFilename, priorities })
  }

  const isValid =
    address.trim() !== "" &&
    ingestedFilename !== "" &&
    priorities.length > 0

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-8">
      {/* Step 1: Address */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground text-xs font-semibold">
            1
          </div>
          <Label
            htmlFor="address"
            className="text-sm font-semibold tracking-wide uppercase text-foreground"
          >
            Property Address
          </Label>
        </div>
        <div className="relative">
          <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="address"
            placeholder="123 Main St, City, State 12345"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            className="pl-10 h-11 bg-card border-border"
          />
        </div>
      </div>

      {/* Step 2: Inspection Report */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground text-xs font-semibold">
            2
          </div>
          <Label className="text-sm font-semibold tracking-wide uppercase text-foreground">
            Inspection Report
          </Label>
        </div>

        {/* PDF upload — indexed into RAG for analysis */}
        <div className="flex flex-col gap-3">
            {/* File upload area */}
            {uploadStatus === "idle" || uploadStatus === "error" ? (
              <label
                htmlFor="pdf-upload"
                className="group relative flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-border bg-card p-8 cursor-pointer transition-colors hover:border-primary/40 hover:bg-secondary/50"
              >
                <input
                  ref={fileInputRef}
                  id="pdf-upload"
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={handleFileChange}
                  className="sr-only"
                />
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-secondary">
                  <Upload className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-foreground">
                    Drop your inspection PDF here
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    or click to browse. Max 10MB.
                  </p>
                </div>
              </label>
            ) : uploadStatus === "uploading" ? (
              <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-border bg-card p-8">
                <Loader2 className="h-8 w-8 text-primary animate-spin" />
                <div className="text-center">
                  <p className="text-sm font-medium text-foreground">
                    Processing PDF...
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Extracting text from {uploadedFile?.name}
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3 rounded-lg border border-border bg-card p-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent/10">
                  <File className="h-5 w-5 text-accent" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {uploadedFile?.name}
                  </p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <CheckCircle2 className="h-3.5 w-3.5 text-accent" />
                    <p className="text-xs text-muted-foreground">
                    {pdfPages} {pdfPages === 1 ? "page" : "pages"}, {chunksIngested} chunks indexed
                    </p>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={handleRemoveFile}
                  className="h-8 w-8 shrink-0 text-muted-foreground hover:text-destructive"
                >
                  <X className="h-4 w-4" />
                  <span className="sr-only">Remove file</span>
                </Button>
              </div>
            )}

            {uploadStatus === "error" && (
              <div className="flex items-center gap-2 rounded-md bg-destructive/10 px-3 py-2">
                <AlertCircle className="h-4 w-4 shrink-0 text-destructive" />
                <p className="text-xs text-destructive">{uploadError}</p>
              </div>
            )}

        </div>
      </div>

      {/* Step 3: User preferences while buying property */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground text-xs font-semibold">
            3
          </div>
          <Label className="text-sm font-semibold tracking-wide uppercase text-foreground">
            Your preferences while buying this property
          </Label>
          <span className="text-xs text-muted-foreground ml-auto">
            {priorities.length}/10 selected
          </span>
        </div>
        <p className="text-xs text-muted-foreground -mt-1">
          Select what matters most to you — analysis will prioritize these areas.
        </p>

        {/* Selected priorities */}
        {priorities.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {priorities.map((p) => (
              <Badge
                key={p}
                variant="default"
                className="gap-1 pr-1 bg-primary text-primary-foreground"
              >
                {p}
                <button
                  type="button"
                  onClick={() => removePriority(p)}
                  className="ml-1 rounded-full p-0.5 hover:bg-primary-foreground/20 transition-colors"
                  aria-label={`Remove ${p}`}
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}

        {/* Options grouped by category */}
        <div className="flex flex-col gap-4">
          {PRIORITY_CATEGORIES.map((category) => {
            const availableItems = category.items.filter(
              (p) => !priorities.includes(p)
            )
            if (availableItems.length === 0) return null
            return (
              <div key={category.label} className="flex flex-col gap-2">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  {category.label}
                </span>
                <div className="flex flex-wrap gap-2">
                  {availableItems.map((priority) => (
                    <button
                      key={priority}
                      type="button"
                      onClick={() => togglePriority(priority)}
                      disabled={priorities.length >= 10}
                      className="rounded-md border border-border bg-card px-3 py-1.5 text-sm text-foreground hover:bg-secondary hover:border-primary/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {priority}
                    </button>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {/* Custom priority */}
        <div className="flex gap-2">
          <Input
            placeholder="Add a custom priority..."
            value={customPriority}
            onChange={(e) => setCustomPriority(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault()
                e.stopPropagation()
                addCustomPriority()
              }
            }}
            className="h-9 bg-card border-border text-sm"
            disabled={priorities.length >= 10}
            autoComplete="off"
            aria-label="Add custom priority"
          />
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={addCustomPriority}
            disabled={!customPriority.trim() || priorities.length >= 10}
            className="shrink-0"
          >
            Add
          </Button>
        </div>
      </div>

      {/* Submit */}
      <Button
        type="submit"
        disabled={!isValid || isLoading}
        className="h-12 text-base font-semibold gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
      >
        {isLoading ? (
          "Analyzing..."
        ) : (
          <>
            Analyze Inspection Report
            <ArrowRight className="h-4 w-4" />
          </>
        )}
      </Button>
    </form>
  )
}
