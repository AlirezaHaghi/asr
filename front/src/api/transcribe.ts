export interface TranscriptionResult {
  transcription: string
}

export async function transcribeAudio(file: File): Promise<TranscriptionResult> {
  const form = new FormData()
  form.append('audio', file)
  const res = await fetch('http://localhost:8000/transcribe', {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    throw new Error(`Request failed — ${res.status} ${res.statusText}`)
  }

  return res.json()
}