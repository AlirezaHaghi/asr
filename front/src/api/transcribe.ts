const API_BASE = 'http://localhost:8000'
// const API_BASE = process.env.API_URL ?? 'http://localhost:8000'

export interface TranscriptionResult {
  transcription: string
  accent: string | null
  confidence: number | null
  accent_confidence: number | null
}

export async function transcribeAudio(file: File): Promise<TranscriptionResult> {
  const form = new FormData()
  form.append('audio', file)

  const res = await fetch(`${API_BASE}/transcribe`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    throw new Error(`Request failed — ${res.status} ${res.statusText}`)
  }

  return res.json()
}
